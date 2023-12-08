import os
import builtins
import torch
import random
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils

import warnings

# Filter warnings from the albumentations module
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
import numpy as np
from .tools import network_common as nn_common
from .tools.Dataset_confidence_map import Dataset

TESTING_MODE = True


def load_checkpoint(result_path=None, node_name=None, distributed=False, device_id_str=None,
                    nn_param=None):
    assert result_path is not None, "result_path is None"
    if os.path.isfile(os.path.join(result_path, 'current.pth')):
        checkpoint_model = torch.load(os.path.join(result_path, 'current.pth'))
    else:
        checkpoint_model = None
        assert nn_param is not None, "nn_param is None"

    if checkpoint_model is not None:
        print(f'[{node_name}]::Loaded checkpoint from', result_path)
    else:
        print(f'[{node_name}]::No checkpoint found in', result_path)

    print(f'[{node_name}]::' + "initializing model")
    # build model
    if checkpoint_model is None:
        base_model = smp.Unet(
            encoder_name=nn_param["encoder"],
            encoder_weights=nn_param["encoder_weight"],
            encoder_depth=nn_param["encoder_depth"],
            classes=len(nn_param["classes"]),
            activation=nn_param["activation"],
        )
    else:
        base_model = checkpoint_model
    if distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        # torch.cuda.set_device(training_param["device_type"][0])
        # model.cuda(training_param["device_type"][0])
        gpu_id = int(os.environ["LOCAL_RANK"])
        model = base_model.to(gpu_id)
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[device_id_str])
        model_without_ddp = model.module
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    print(f'[{node_name}]::' + "model initialized")

    preprocessing_fn = smp.encoders.get_preprocessing_fn(nn_param['encoder'], nn_param['encoder_weight'])

    return model, preprocessing_fn


def initialize(
        model=None,
        preprocessing_fn=None,
        network_img_size=None,
        classes=None,
        distributed=False,
        global_world_size=None,
        local_rank=-1,
        global_rank=-1,
        node_name=None,
        training_param=None,
        dataset_path=None,
):
    print("initialization on:", node_name)
    device_type = training_param['device_type']
    device_id = training_param['device_id']
    device_id_str = training_param['device_id_str']

    x_train_dir = os.path.join(dataset_path, 'train')
    y_train_dir = os.path.join(dataset_path, 'train_mask')
    x_valid_dir = os.path.join(dataset_path, 'val')
    y_valid_dir = os.path.join(dataset_path, 'val_mask')


    loss = smp_utils.losses.MSELoss()
    loss.to(device_id_str)
    metrics = [smp_utils.metrics.IoU(threshold=0.5)]
    optimizer = torch.optim.Adam(model.parameters(), lr=training_param["lr"],
                                 weight_decay=training_param["weight_decay"])

    ### data ###
    train_dataset = Dataset(x_train_dir, y_train_dir, network_img_size, max_id=100 if TESTING_MODE else None,
                            preprocessing=nn_common.get_preprocessing(preprocessing_fn),
                            classes=classes)

    val_dataset = Dataset(x_valid_dir, y_valid_dir, network_img_size, max_id=40 if TESTING_MODE else None,
                          preprocessing=nn_common.get_preprocessing(preprocessing_fn),
                          classes=classes)
    print('the number of image/label in the train: ', len(train_dataset))

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset, batch_size=training_param['batch_size'], shuffle=(train_sampler is None),
        sampler=train_sampler, drop_last=True, pin_memory=True)

    val_sampler = None
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, pin_memory=True,
    )

    return {
        "model": model,
        "loss": loss,
        "metrics": metrics,
        "optimizer": optimizer,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "distributed": distributed,
        'device_id': device_id,
        'device_type': device_type,
        "device_id_str": device_id_str,
    }


def assert_device_type(device_src, device_dst_str, info):
    if str(device_src) != device_dst_str:
        print(f"{info}: {device_src} but got {device_dst_str}")
        raise RuntimeError(f"{info} not initialized correctly")


def _train_one_epoch(
        model=None,
        loss=None,
        # metrics=None,
        optimizer=None,
        train_loader=None,
        device_id=None,
        global_rank=-1,
        device_id_str=None,
):
    model.train()
    torch.cuda.set_device(device_id_str)
    # loss_meter = smp_utils.train.AverageValueMeter()
    # metrics_meters = {metric.__name__: smp_utils.train.AverageValueMeter() for metric in metrics}
    dist.barrier()

    # Log the device of model, optimizer, and loss
    assert_device_type(next(model.parameters()).device, device_id_str, "Model device")
    assert_device_type(loss.device if hasattr(loss, 'device') else device_id_str, device_id_str, "Loss device")

    # Logging the device of model parameters
    for name, param in model.named_parameters():
        assert_device_type(param.device, device_id_str, f"Parameter {name} device")

    num_batches = len(train_loader)
    for i, (x, y) in enumerate(train_loader):
        if global_rank == 0 and i % 2 == 0:
            print("train progress: ", i, "/", num_batches, "batches")

        loss_val, y_pred = _train_batch_update(model.to(device_id_str), optimizer, loss.to(device_id_str),
                                               x.to(device_id_str), y.to(device_id_str),
                                               defensive=True, device_id_str=device_id_str, global_rank=global_rank)

    dist.barrier()


def _train_batch_update(model, optimizer, loss, x_, y_, defensive=False, device_id_str=None, global_rank=-1):
    if defensive:
        print(global_rank, type(model))
        print(global_rank, device_id_str)
        for i in model.named_parameters():
            if str(i[1].device) != device_id_str:
                print(f"grank-{global_rank}, {i[0]} -> {i[1].device} <-> {device_id_str}")
                raise RuntimeError("model not initialized correctly")
        if str(x_.device) != device_id_str:
            raise RuntimeError("x not initialized correctly")
        if str(y_.device) != device_id_str:
            raise RuntimeError("y not initialized correctly")
    optimizer.zero_grad()
    prediction = model.forward(x_)
    loss_val_ = loss(prediction, y_)
    loss_val_.backward()
    optimizer.step()
    return loss_val_, prediction


def _valid_one_epoch(
        model=None, loss=None, metrics=None, val_loader=None, device_id=None, device_id_str=None
):
    model.eval()
    torch.cuda.set_device(device_id)

    loss_meter = smp_utils.train.AverageValueMeter()
    metrics_meters = {metric.__name__: smp_utils.train.AverageValueMeter() for metric in metrics}
    logs = {}

    def batch_update(model_, x_, y_):
        with torch.no_grad():
            prediction = model_.forward(x_)
            loss_val_ = loss(prediction, y_)
        return loss_val_, prediction

    num_batches = len(val_loader)
    for i, (x, y) in enumerate(val_loader):
        if i % 10 == 0:
            print("val progress: ", i, "/", num_batches, "batches")

        x, y = x.to(device_id_str), y.to(device_id_str)
        loss_val, y_pred = batch_update(model, x, y)

        # update loss logs
        loss_value = loss_val.cpu().detach().numpy()
        loss_meter.add(loss_value)
        loss_logs = {loss.__name__: loss_meter.mean}
        logs.update(loss_logs)

        # update metrics logs
        for metric_fn in metrics:
            metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
            metrics_meters[metric_fn.__name__].add(metric_value)
        metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
        logs.update(metrics_logs)

        return logs


def train(
        model=None,
        loss=None,
        metrics=None,
        optimizer=None,
        result_path=None,
        train_loader=None,
        val_loader=None,
        distributed=False,
        start_epoch=0,
        epochs=0,
        global_rank=0,
        device_id=None,
        device_type=None,
        device_id_str=None,
        **kwargs
):
    # train_epoch = smp_utils.train.TrainEpoch(model, loss=loss, metrics=metrics, optimizer=optimizer, device=device,
    #                                          verbose=True)
    # valid_epoch = smp_utils.train.ValidEpoch(model, loss=loss, metrics=metrics, device=device, verbose=True)
    print("device_id:", device_id)
    torch.cuda.set_device(device_id)
    ### main loop ###
    max_score = 1
    for epoch in range(start_epoch, epochs):
        np.random.seed(epoch)
        random.seed(epoch)
        # fix sampling seed such that each gpu gets different part of dataset
        if distributed:
            train_loader.sampler.set_epoch(epoch)
        # adjust lr if needed #
        # train_epoch.run(train_loader)
        _train_one_epoch(model=model, loss=loss, optimizer=optimizer, train_loader=train_loader,
                         device_id=device_id, device_id_str=device_id_str,
                         global_rank=global_rank)
        if global_rank == 0:  # only operator on rank 0 master node
            # valid_logs = valid_epoch.run(val_loader)
            valid_logs = _valid_one_epoch(model=model, loss=loss, metrics=metrics, val_loader=val_loader,
                                          device_id=device_id, device_id_str=device_id_str)
            # save checkpoint if needed #
            torch.save(model, os.path.join(result_path, 'current.pth'))
            if max_score > valid_logs['mse_loss']:
                max_score = valid_logs['mse_loss']
                torch.save(model, os.path.join(result_path, 'best_model.pth'))
                print('Best Model saved! on epoch: ', epoch)

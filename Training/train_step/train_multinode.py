import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

##################################################################
# Example of using segmentation_models_pytorch Unet
# import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils
# from .tools.Dataset_confidence_map import Dataset
# from .tools import network_common as nn_common
##################################################################
from .tools.Dataset import ExampleDataset

TESTING_MODE = False


def ddp_setup(backend: str = "nccl"):
    init_process_group(backend=backend)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def ddp_finalize():
    destroy_process_group()

class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            val_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            save_every: int,
            snapshot_path: str,
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets, loss):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = loss(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch, loss):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            self._run_batch(source, targets, loss)

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def _val_batch_update(self, loss, x_, y_):
        with torch.no_grad():
            prediction = self.model.forward(x_)
            loss_val_ = loss(prediction, y_)
        return loss_val_, prediction

    def _valid_one_epoch(self, loss: torch.nn.MSELoss = None, metrics=None):
        self.model.eval()
        loss_meter = smp_utils.train.AverageValueMeter()
        metrics_meters = {metric.__name__: smp_utils.train.AverageValueMeter() for metric in metrics}
        logs = {}

        num_batches = len(self.val_data)
        for i, (x, y) in enumerate(self.val_data):
            if i % 50 == 0:
                print("val progress: ", i, "/", num_batches, "batches")
            x = x.to(self.local_rank)
            y = y.to(self.local_rank)
            loss_val, y_pred = self._val_batch_update(loss, x, y)

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

        self.model.train()
        return logs

    def train(self, max_epochs: int, best_save_path: str = None, val_params: dict = None, loss=None, ):
        max_score = 1
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch, loss)
            if self.global_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
                if val_params is not None and best_save_path is not None:
                    valid_logs = self._valid_one_epoch(
                        loss=loss,
                        metrics=val_params['metrics'],
                    )
                    print("current loss: ", valid_logs['mse_loss'], "min loss: ", max_score)
                    if max_score > valid_logs['mse_loss']:
                        max_score = valid_logs['mse_loss']
                        torch.save(self.model, best_save_path)
                        print('Best Model saved! on epoch: ', epoch)
        if self.global_rank == 0 and (not os.path.exists(best_save_path)):  # save the model if it is not saved yet
            torch.save(self.model, best_save_path)


def load_train_objs(
        result_path=None,
        node_name=None,
        nn_param=None,
        optimizer_param=None,
        dataset_path=None,
):
    if result_path is not None and os.path.isfile(result_path):
        checkpoint_model = torch.load(result_path)
    else:
        checkpoint_model = None
        assert nn_param is not None, "nn_param is None"

    if checkpoint_model is None:
        base_model = torch.nn.Linear(20, 1)  # load your model
        ##################################################################
        # Example of using segmentation_models_pytorch Unet
        # base_model = smp.Unet(
        #     encoder_name=nn_param["encoder"],
        #     encoder_weights=nn_param["encoder_weight"],
        #     encoder_depth=nn_param["encoder_depth"],
        #     classes=len(nn_param["classes"]),
        #     activation=nn_param["activation"],
        # )
        ##################################################################
    else:
        base_model = checkpoint_model

    ##################################################################
    # Example of from segmentation_models_pytorch Unet
    # x_train_dir = os.path.join(dataset_path, 'train')
    # y_train_dir = os.path.join(dataset_path, 'train_mask')
    # x_valid_dir = os.path.join(dataset_path, 'val')
    # y_valid_dir = os.path.join(dataset_path, 'val_mask')
    # preprocessing_fn = smp.encoders.get_preprocessing_fn(nn_param['encoder'], nn_param['encoder_weight'])
    # train_dataset = Dataset(x_train_dir, y_train_dir,
    #                         nn_param['network_img_size'], max_id=100 if TESTING_MODE else None,
    #                         preprocessing=nn_common.get_preprocessing(preprocessing_fn),
    #                         classes=nn_param["classes"])
    #
    # val_dataset = Dataset(x_valid_dir, y_valid_dir,
    #                       nn_param['network_img_size'], max_id=40 if TESTING_MODE else None,
    #                       preprocessing=nn_common.get_preprocessing(preprocessing_fn),
    #                       classes=nn_param["classes"])
    # optimizer = torch.optim.Adam(base_model.parameters(), lr=optimizer_param["lr"],
    #                              weight_decay=optimizer_param["weight_decay"])
    ##################################################################

    train_dataset = ExampleDataset(2048)
    val_dataset = ExampleDataset(512)

    optimizer = torch.optim.SGD(base_model.parameters(), lr=1e-3)

    return train_dataset, val_dataset, base_model, optimizer


def prepare_dataloader(train_dataset: torch.utils.data.Dataset,
                       val_dataset: torch.utils.data.Dataset,
                       batch_size: int):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(train_dataset)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, pin_memory=True,
    )
    return train_loader, val_loader

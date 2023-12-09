import os
import builtins
import argparse
# import torch
# import torch.distributed as dist
from train_step.train_multinode import ddp_setup, load_train_objs, prepare_dataloader, Trainer, ddp_finalize
##################################################################
# Example of using segmentation_models_pytorch Unet
import segmentation_models_pytorch.utils as smp_utils
import segmentation_models_pytorch as smp
##################################################################


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument("--weight_decay", default=1e-5, type=float, help="weight decay")
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=10, type=int, help='batch size per GPU')
    parser.add_argument('--save_every', default=2, type=int, help='checkpoint frequency')

    # other arguments
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    try:
        node_name = os.environ['SLURMD_NODENAME']
    except KeyError:
        node_name = 'local'
    _args = parse_args()

    ##################################################################
    # Example of using segmentation_models_pytorch Unet
    # from config.training_param import TrainingParameters
    # from config.network_param import NetworkParameter
    # nn_param_override = {
    #     "CLASSES": ['class_label1', 'class_label2']
    # }
    # train_param_override = {
    #     "model": 'xxx_model',
    #     #############################
    #     # Note 1:
    #     # 'dir_base_path': 'storage/user_data/[YOUR DATA DIRECTORY]'
    #     'dir_base_path': 'storage/user_data/[YOUR DATA DIRECTORY]'
    #     #############################
    #
    # }
    # train_param = TrainingParameters(**train_param_override)
    # nn_param = NetworkParameter(**nn_param_override)
    # train_param.re_process_path()
    # result_path = train_param.result_path
    ##################################################################
    result_path = './test_run/'

    try:
        global_world_size = int(os.environ["WORLD_SIZE"])
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
    except KeyError:
        global_world_size = 1
        local_world_size = 1
        local_rank = 0
        global_rank = 0

    ######### file io should only run on rank 0 #########
    if global_rank == 0:
        os.makedirs(result_path, exist_ok=True)

    print(f'[{node_name}]::World size:', global_world_size)
    print(f'[{node_name}]::Number of GPU avaliable:',
          global_world_size, "-> on local rank", local_rank, "global rank", global_rank)

    ##########################################
    # training code start here
    ##########################################
    # distributed data parallel environment + process setup
    ddp_setup()
    print(f'{node_name}::Done DDP setup')

    ##################################################################
    # Example of using segmentation_models_pytorch Unet
    # loss and metrics
    # loss = smp_utils.losses.MSELoss()
    # loss.cuda(int(os.environ["LOCAL_RANK"]))
    # metrics = [smp_utils.metrics.IoU(threshold=0.5)]
    ##################################################################

    # load dataset and model
    ##################################################################
    # you will need to customise this function to load your own dataset and model
    ##################################################################
    train_dataset, val_dataset, model, optimizer = load_train_objs(
        # dataset_path=train_param.dataset_path,
        # result_path=os.path.join(train_param.result_path, 'best_model.pth'),
        node_name=node_name,
        # nn_param={
        #     "classes": nn_param.CLASSES,
        #     "encoder": nn_param.ENCODER,
        #     "encoder_weight": nn_param.ENCODER_WEIGHTS,
        #     "activation": nn_param.ACTIVATION,
        #     "network_img_size": train_param.network_img_size,
        #     'encoder_depth': 5,
        # },
        # optimizer_param={
        #     'lr': _args.lr,
        #     'weight_decay': _args.weight_decay,
        # },
    )
    print(f'{node_name}::GPU{global_rank}::train dataset size: {len(train_dataset)}')
    print(f'{node_name}::GPU{global_rank}::validation dataset size: {len(val_dataset)}')
    print(f'{node_name}::GPU{global_rank}::Done loading dataset')

    train_loader, val_loader = prepare_dataloader(train_dataset, val_dataset,
                                                  _args.batch_size)
    print(f'{node_name}::GPU{global_rank}::Done preparing dataloader: batch size {_args.batch_size}')
    trainer = Trainer(
        model=model,
        train_data=train_loader, val_data=val_loader,
        optimizer=optimizer, save_every=_args.save_every,
        snapshot_path=os.path.join(result_path, 'snapshots.save'),
    )
    print(f'{node_name}::GPU{global_rank}::Trainer initialised')

    print(f'{node_name}::GPU{global_rank}::Start train')
    trainer.train(_args.epochs,
                  best_save_path=os.path.join(result_path, 'best_model.pth'),
                  ##################################################################
                  # Example of using segmentation_models_pytorch Unet
                  # val_params={
                  #     'metrics': metrics,
                  # },
                  # loss=loss
                  ##################################################################
                  )

    print(f'{node_name}::GPU{global_rank}::End train')
    ddp_finalize()

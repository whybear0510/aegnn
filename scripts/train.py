import argparse
import datetime
import os
import pytorch_lightning as pl
import pytorch_lightning.loggers
import wandb
import yaml

import aegnn
import torch_geometric
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Model name to train.")
    parser.add_argument("--task", type=str, required=True, help="Task to perform, such as detection or recognition.")
    parser.add_argument("--dim", type=int, help="Dimensionality of input data", default=3)
    parser.add_argument("--seed", default=12345, type=int)

    group = parser.add_argument_group("Trainer")
    group.add_argument("--max-epochs", default=100, type=int)
    group.add_argument("--min-epochs", default=35, type=int)
    group.add_argument("--overfit-batches", default=0.0, type=int)
    group.add_argument("--log-every-n-steps", default=10, type=int)
    group.add_argument("--gradient_clip_val", default=0.0, type=float)
    group.add_argument("--limit_train_batches", default=1.0, type=int)
    group.add_argument("--limit_val_batches", default=1.0, type=int)

    parser.add_argument("--log-gradients", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cpu", action="store_true") # default is NOT using cpu
    # parser.add_argument("--gpu", default=None, type=int)

    parser.add_argument("--init-lr", default=0.001, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--wandb-sweep", action="store_true")
    parser.add_argument("--act", default='relu', type=str)
    parser.add_argument("--run-name", default=None, type=str)
    parser.add_argument("--grid-div", default=4, type=int)
    parser.add_argument("--conv-type", default='spline', type=str)

    #copy from flops.py
    parser.add_argument("--radius", default=3.0, help="radius of radius graph generation")
    parser.add_argument("--max-num-neighbors", default=32, help="max. number of neighbors in graph")
    parser.add_argument("--pooling-size", default=(10, 10))

    parser = aegnn.datasets.EventDataModule.add_argparse_args(parser)
    return parser.parse_args()


def main(args):

    log_dir = os.environ["AEGNN_LOG_DIR"]
    loggers = [aegnn.utils.loggers.LoggingLogger(None if args.debug else log_dir, name="debug")]
    project = f"aegnn-{args.dataset}-{args.task}"
    experiment_name = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if args.run_name is None:
        runs_name = f'lr{args.init_lr}_{args.conv_type}_{experiment_name}'
    else:
        runs_name = f'lr{args.init_lr}_{args.conv_type}_{args.run_name}_{experiment_name}'

    dm = aegnn.datasets.by_name(args.dataset).from_argparse_args(args)
    dm.setup()

    train_args = {
        'max_epochs': args.max_epochs,
        'learning_rate': args.init_lr,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size
    }

    model_args = {
        'network': args.model,
        'dataset': args.dataset,
        'num_classes': dm.num_classes,
        'img_shape': dm.dims,
        'dim': args.dim,
        'learning_rate': train_args["learning_rate"],
        'weight_decay': train_args["weight_decay"],
        'bias': True,
        'root_weight': True,
        'act': args.act,
        'grid_div': args.grid_div,
        'conv_type': args.conv_type
    }

    model = aegnn.models.by_task(args.task)(**model_args)

    # training the async model (?), copy from flop.py
    # edge_attr = torch_geometric.transforms.Cartesian(cat=False, max_value=10.0)
    # model = aegnn.asyncronous.make_model_asynchronous(model, args.radius, list(dm.dims), edge_attr)


    wandb.init(project="aegnn", entity="yyfteam", name=runs_name)
    # log_settings = wandb.Settings(start_method="thread")  # for Windows?

    if not args.debug:
        wandb_logger = pl.loggers.WandbLogger(project=project, save_dir=log_dir)
        if args.log_gradients:
            wandb_logger.watch(model, log="gradients")  # gradients plot every 100 training batches
        loggers.append(wandb_logger)


    logger = pl.loggers.LoggerCollection(loggers)

    checkpoint_path = os.path.join(log_dir, "checkpoints", args.dataset, args.task, experiment_name)
    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        # pl.callbacks.EarlyStopping(monitor="Val/Accuracy", min_delta=0.01, patience=8, verbose=False, mode="max"),
        aegnn.utils.callbacks.BBoxLogger(classes=dm.classes),
        aegnn.utils.callbacks.PHyperLogger(args),
        aegnn.utils.callbacks.EpochLogger(),
        aegnn.utils.callbacks.FileLogger([model, model.model, dm]),
        aegnn.utils.callbacks.FullModelCheckpoint(dirpath=checkpoint_path)
    ]


    trainer_kwargs = dict()

    trainer_kwargs["profiler"] = "simple" if args.profile else False
    trainer_kwargs["weights_summary"] = "full"
    trainer_kwargs["track_grad_norm"] = 2 if args.log_gradients else -1
    # trainer_kwargs["accumulate_grad_batches"] = 64
    trainer_kwargs["max_epochs"] = train_args['max_epochs']
    trainer_kwargs["logger"] = logger
    trainer_kwargs["callbacks"] = callbacks

    # trainer_kwargs["gpus"] = [args.gpu] if args.gpu is not None else None
    if not args.cpu:
        trainer_kwargs["accelerator"] = "gpu"
        trainer_kwargs["devices"] = -1
        trainer_kwargs["auto_select_gpus"] = True
    else:
        trainer_kwargs["accelerator"] = "cpu"


    trainer = pl.Trainer.from_argparse_args(args, **trainer_kwargs)
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())


if __name__ == '__main__':
    arguments = parse_args()
    pl.seed_everything(arguments.seed)
    main(arguments)

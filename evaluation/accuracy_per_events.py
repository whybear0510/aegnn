import aegnn
import argparse
import logging
import os
import numpy as np
import pandas as pd
import torch
import pytorch_lightning.metrics.functional as pl_metrics

from torch_geometric.data import Batch
from torch_geometric.utils import subgraph
from tqdm import tqdm
from typing import Iterable, Tuple


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file", help="Path of model to evaluate.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fast-test", action="store_true")
    # copy from flops.py
    parser.add_argument("--radius", default=3.0, help="radius of radius graph generation")
    parser.add_argument("--max-num-neighbors", default=32, help="max. number of neighbors in graph")
    parser.add_argument("--pooling-size", default=(10, 10))

    parser = aegnn.datasets.EventDataModule.add_argparse_args(parser)
    return parser.parse_args()


def sample_batch(batch_idx: torch.Tensor, num_samples: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Samples a subset of graphs in a batch and returns the sampled nodes and batch_idx.

    >> batch = torch.from_numpy(np.random.random_integers(0, 10, size=100))
    >> batch = torch.sort(batch).values
    >> sample_batch(batch, max_num_events=2)
    """
    subset = []
    subset_batch_idx = []
    for i in torch.unique(batch_idx):
        batch_idx_i = torch.nonzero(torch.eq(batch_idx, i)).flatten()
        # sample_idx_i = torch.randperm(batch_idx_i.numel())[:num_samples]
        # subset.append(batch_idx_i[sample_idx_i])
        sample_idx_i = batch_idx_i[:num_samples]
        subset.append(sample_idx_i)
        subset_batch_idx.append(torch.ones(sample_idx_i.numel()) * i)
    return torch.cat(subset).long(), torch.cat(subset_batch_idx).long()


def evaluate(model, data_loader: Iterable[Batch], max_num_events: int) -> float:
    accuracy = []
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    for i, batch in enumerate(data_loader):
        batch_idx = getattr(batch, 'batch')
        subset, subset_batch_idx = sample_batch(batch_idx, num_samples=max_num_events)
        is_in_subset = torch.zeros(batch_idx.numel(), dtype=torch.bool)
        is_in_subset[subset] = True

        edge_index, edge_attr = subgraph(is_in_subset, batch.edge_index, edge_attr=batch.edge_attr, relabel_nodes=True)
        sample = Batch(x=batch.x[is_in_subset, :], pos=batch.pos[is_in_subset, :], y=batch.y,
                       edge_index=edge_index, edge_attr=edge_attr, batch=subset_batch_idx)
        logging.debug(f"Done data-processing, resulting in {sample}")

        # copy from flops.py
        # input_shape = torch.tensor([*dm.dims, sample.pos.shape[-1]], device=device)
        # model_ = aegnn.models.networks.GraphRes(dm.name, input_shape, dm.num_classes, pooling_size=args.pooling_size)
        # model_.to(device)
        # model = aegnn.asyncronous.make_model_asynchronous(model, args.radius, list(dm.dims), edge_attr, **kwargs)

        # original
        sample = sample.to(model.device)

        outputs_i = model.forward(sample)
        y_hat_i = torch.argmax(outputs_i, dim=-1)

        accuracy_i = pl_metrics.accuracy(preds=y_hat_i, target=sample.y).cpu().numpy()
        accuracy.append(accuracy_i)
    return float(np.mean(accuracy))


def main(args, model, data_module):
    if args.fast_test:
        max_num_events = [1000, 2000, 2500, 4000, 5000, 8000, 10000]
    else:
        max_num_events = np.arange(1000, 10000, step=200)
    df = pd.DataFrame()

    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "aegnn_results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "accuracy_per_events.pkl")

    for max_count in tqdm(max_num_events):
        data_loader = data_module.val_dataloader(num_workers=16).__iter__()
        # data_loader = data_module.test_dataloader(num_workers=16).__iter__()
        accuracy = evaluate(model, data_loader, max_num_events=max_count)
        logging.debug(f"Evaluation with max_num_events = {max_count} => Recognition accuracy = {accuracy}")

        df = df.append({"accuracy": accuracy, "max_num_events": max_count}, ignore_index=True)
        df.to_pickle(output_file)

    print(f"Results are logged in {output_file}")
    return df


if __name__ == '__main__':
    args = parse_args()
    if args.debug:
        _ = aegnn.utils.loggers.LoggingLogger(None, name="debug")

    model_eval = torch.load(args.model_file).to(args.device)
    dm = aegnn.datasets.by_name(args.dataset).from_argparse_args(args)
    dm.setup()
    # model_eval = None

    main(args, model_eval, dm)

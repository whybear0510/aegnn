import aegnn
import argparse
import logging
import os
import numpy as np
import pandas as pd
import torch
import torchmetrics.functional as pl_metrics
import torch_geometric

from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from tqdm import tqdm
from typing import Iterable, Tuple

import copy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file", help="Path of model to evaluate.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fast-test", action="store_true")
    # copy from flops.py
    parser.add_argument("--radius", default=3.0, help="radius of radius graph generation")
    parser.add_argument("--max-num-neighbors", default=32, help="max. number of neighbors in graph")
    parser.add_argument("--pooling-size", default=(12, 10))

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

def sample_initial_data(sample, num_events: int, radius: float, edge_attr, max_num_neighbors: int):
    data = Data(x=sample.x[:num_events], pos=sample.pos[:num_events])
    data.batch = torch.zeros(data.num_nodes, device=data.x.device)
    data.edge_index = torch_geometric.nn.radius_graph(data.pos, r=radius, max_num_neighbors=max_num_neighbors).long()
    if edge_attr is not None: data.edge_attr = edge_attr(data).edge_attr
    nxt_event_idx = num_events
    return data, nxt_event_idx

def sample_new_data(sample, nxt_event_idx):
    x_new = sample.x[nxt_event_idx, :].view(1, -1)
    pos_new = sample.pos[nxt_event_idx, :3].view(1, -1)  #TODO: :2 ? no time?
    event_new = Data(x=x_new, pos=pos_new, batch=torch.zeros(1, dtype=torch.long))
    nxt_event_idx += 1
    return event_new, nxt_event_idx



def evaluate(model, data_loader: Iterable[Batch], args, img_size, init_event: int = None, iter_cnt: int = None) -> float:
    predss = []
    targets = []

    edge_attr = torch_geometric.transforms.Cartesian(cat=False, max_value=10.0)
    # edge_attr = None # for GCN

    # async_model = aegnn.asyncronous.make_model_asynchronous(model, args.radius, img_size, edge_attr)





    for i, batch in enumerate(tqdm(data_loader, position=1)):

        model_i = torch.load(args.model_file).to(args.device)
        model_i.eval()
        async_model = aegnn.asyncronous.make_model_asynchronous(model_i, args.radius, img_size, edge_attr)
        #TODO: theoretically, I should translate origin model to async one outside the loop.
        #      but it will cause a data fault when read the 2nd sample and async_model processes it.
        #      and I also dont know how to deep copy a model variable

        sample = batch


        sample = sample.to(model.device)
        tot_nodes = sample.num_nodes
        # print(f'sample.num_nodes = {sample.num_nodes}')
        # outputs_i = model.forward(sample)
        # y_hat_i = torch.argmax(outputs_i, dim=-1)
        # predss.append(y_hat_i)
        targets.append(sample.y)
        # print(f'ground truth = {sample.y}')

        if init_event is None and iter_cnt is not None:
            init_num_event = tot_nodes - iter_cnt
        elif init_event is not None:
            init_num_event = init_event
        else:
            init_num_event = tot_nodes

        sub_predss = []

        events_initial, nxt_event_idx = sample_initial_data(sample, init_num_event, args.radius, edge_attr, args.max_num_neighbors)
        events_initial = events_initial.to(model.device)
        output_init = async_model.forward(events_initial)
        y_init = torch.argmax(output_init, dim=-1)
        sub_predss.append(y_init)

        # print(f'timestamp = {events_initial.pos[-1,2].item()}')

        # print(f'output_init = {output_init}')
        # print(f'y_init = {y_init}')



        cnt = 0
        while nxt_event_idx < sample.num_nodes:
            # print(f'nxt_event_idx = {nxt_event_idx}')
            event_new, nxt_event_idx = sample_new_data(sample, nxt_event_idx)
            event_new = event_new.to(model.device)
            # print(f'timestamp = {event_new.pos[-1,2].item()}')
            output_new = async_model.forward(event_new)
            y_new = torch.argmax(output_new, dim=-1)
            sub_predss.append(y_new)
            # print(f'output_new = {output_new}')
            cnt += 1

        # print(f'cnt={cnt}')

        sub_preds = torch.cat(sub_predss).view(-1,1)
        # print(sub_preds)


        predss.append(sub_preds)

        del events_initial, event_new, model_i, init_num_event



    preds = torch.cat(predss, dim=1)
    target = torch.cat(targets)
    tot_accuracies = []
    for j in range(preds.shape[0]):
        tot_accuracy = pl_metrics.accuracy(preds=preds[j,:], target=target)
        tot_accuracy = tot_accuracy.item()
        tot_accuracies.append(tot_accuracy)
    return tot_accuracies


def main(args, model, data_module):
    # if args.fast_test:
    #     max_num_events = [25000]
    #     # max_num_events = [1000, 2000, 2500, 4000, 5000, 8000, 10000]
    # else:
    #     max_num_events = np.arange(1000, 15000, step=1000)

    init_num_events = [4070]
    iter_cnts=[50]
    img_size = list(data_module.dims)

    df = pd.DataFrame()
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "aegnn_results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "async_accuracy")

    # for init_num_event in tqdm(init_num_events, position=0):
    #     data_loader = data_module.val_dataloader(num_workers=16).__iter__()

    #     accuracy = evaluate(model, data_loader, args=args, init_event=init_num_event, img_size=img_size)
    #     logging.debug(f"Evaluation with init_num_events = {init_num_event} => Recognition accuracy = {accuracy}")

    #     df = pd.concat([df, pd.DataFrame({"accuracy": accuracy, "init_num_events": init_num_event}, index=[0])], ignore_index=True)
    #     df.to_pickle(output_file+'.pkl')
    #     df.to_csv(output_file+'.csv')

    for iter_cnt in tqdm(iter_cnts, position=0):
        data_loader = data_module.val_dataloader(num_workers=16).__iter__()

        accuracy = evaluate(model, data_loader, args=args, iter_cnt=iter_cnt, img_size=img_size)
        index = ['init']
        index.extend([f'{a}' for a in range(iter_cnt)])


        df = pd.concat([df, pd.Series(accuracy, index)])
        df.to_pickle(output_file+'.pkl')
        df.to_csv(output_file+'.csv')

    print(f"Results are logged in {output_file}.*")
    return df


if __name__ == '__main__':
    args = parse_args()
    if args.debug:
        _ = aegnn.utils.loggers.LoggingLogger(None, name="debug")

    model_eval = torch.load(args.model_file).to(args.device)
    model_eval.eval()
    dm = aegnn.datasets.by_name(args.dataset).from_argparse_args(args)
    dm.setup()

    main(args, model_eval, dm)

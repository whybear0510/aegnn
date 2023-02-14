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

from time import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file", help="Path of model to evaluate.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--debug", action="store_true")
    # copy from flops.py
    parser.add_argument("--radius", default=3.0, help="radius of radius graph generation")
    parser.add_argument("--max-num-neighbors", default=32, help="max. number of neighbors in graph")
    parser.add_argument("--pooling-size", default=(12, 10))

    parser = aegnn.datasets.EventDataModule.add_argparse_args(parser)
    return parser.parse_args()


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


@torch.no_grad()
def evaluate(model, data_loader: Iterable[Batch], args, img_size, init_event: int = None, iter_cnt: int = None) -> float:
    predss = []
    targets = []

    edge_attr = torch_geometric.transforms.Cartesian(cat=False, max_value=10.0)

    async_model = aegnn.asyncronous.make_model_asynchronous(model, args.radius, img_size, edge_attr)
    async_model.eval()


    df = pd.DataFrame()
    output_file = '/users/yyang22/thesis/aegnn_project/aegnn_results/mid_result'

    num_test_samples = 100

    # for NCars dataset, #events: min=500 at 1422, max=40810 at 219; #samples = 2462
    for i, batch in enumerate(tqdm(data_loader, position=1, desc='Samples', total=num_test_samples)):

        if i==num_test_samples: break

        # if i!=9:continue  #!debug


        async_model = aegnn.asyncronous.reset_async_module(async_model)

        sample = batch
        sample = sample.to(model.device)
        tot_nodes = sample.num_nodes
        # targets.append(sample.y)
        # print(f'ground truth = {sample.y}')

        if init_event is None and iter_cnt is not None:
            init_num_event = tot_nodes - iter_cnt
        elif init_event is not None:
            init_num_event = init_event
        else:
            # init_num_event = 3500  # guarantee to have an edge between nodes #!debug from here: 54: reason: after torch.unique, 'size' is lost
            init_num_event = 25

        sub_predss = []


        events_initial, nxt_event_idx = sample_initial_data(sample, init_num_event, args.radius, edge_attr, args.max_num_neighbors)
        events_initial = events_initial.to(model.device)
        if events_initial.edge_index.nelement() == 0:
            continue
        # code_time = []
        # code_time.append(time())
        targets.append(sample.y)
        output_init = async_model.forward(events_initial)
        # code_time.append(time())
        y_init = torch.argmax(output_init, dim=-1)
        sub_predss.append(y_init)

        # print(f'timestamp = {events_initial.pos[-1,2].item()}')

        # code_time_diff = []
        # idx=0
        # while idx < len(code_time)-1:
        #     code_time_diff.append(code_time[idx+1]-code_time[idx])
        #     idx += 1
        # print(f'init:{code_time_diff}')


        with tqdm(total=(tot_nodes-nxt_event_idx), position=2, leave=False, desc='Events') as pbar:
            while nxt_event_idx < tot_nodes:
                torch.cuda.empty_cache()
                # code_time = []
                event_new, nxt_event_idx = sample_new_data(sample, nxt_event_idx)
                event_new = event_new.to(model.device)
                # print(f'timestamp = {event_new.pos[-1,2].item()}')
                # code_time.append(time())
                output_new = async_model.forward(event_new)
                # code_time.append(time())
                y_new = torch.argmax(output_new, dim=-1)
                sub_predss.append(y_new)
                pbar.update(1)

        # targets.append(sample.y)
        # output_init = async_model.forward(sample) #!: TODO: debug: "async_model" have bugs, causing it wrong!
        # # code_time.append(time())
        # y_init = torch.argmax(output_init, dim=-1)
        # sub_predss.append(y_init)



        sub_preds = torch.cat(sub_predss)

        # df = pd.concat([df, pd.Series(sub_preds.cpu())], axis=1)
        # df.columns = [truth.cpu() for truth in targets]
        column_name = pd.MultiIndex.from_tuples([(i, sample.y.cpu().item())], names=['i', 'gnd_truth'])
        df.insert(len(targets)-1, column_name, pd.Series(sub_preds.cpu()), allow_duplicates=True)
        df.to_pickle(output_file+'.pkl')
        df.to_csv(output_file+'.csv')

        predss.append(sub_preds)

        # del events_initial, event_new, init_num_event


        # code_time_diff = []
        # idx=0
        # while idx < len(code_time)-1:
        #     code_time_diff.append(code_time[idx+1]-code_time[idx])
        #     idx += 1
        # print(f'added:{code_time_diff}')



    # "torch.nested" is not yet supported!
    # preds_nt = torch.nested.nested_tensor(predss)
    # preds = torch.nested.to_padded_tensor(preds_nt, 0.0)



    for i, sub_preds in enumerate(predss):
        tmp = torch.nn.functional.pad(sub_preds, (0,41000-len(sub_preds)), value=sub_preds[-1].item())
        predss[i] = tmp.unsqueeze(0)

    preds = torch.cat(predss, dim=0)
    target = torch.cat(targets).unsqueeze(1)
    tot_accuracies = []
    for j in tqdm(range(41000), position=3, desc='Acc Calc', leave=False):
        tot_accuracy = pl_metrics.accuracy(preds=preds[:,j], target=target)
        tot_accuracy = tot_accuracy.item()
        tot_accuracies.append(tot_accuracy)
    return tot_accuracies


def main(args, model, data_module):

    # init_num_events = [4070]
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

    # for iter_cnt in tqdm(iter_cnts, position=0, desc='Tests'):
    #     data_loader = data_module.val_dataloader(num_workers=16).__iter__()

    #     # accuracy = evaluate(model, data_loader, args=args, iter_cnt=iter_cnt, img_size=img_size)
    #     accuracy = evaluate(model, data_loader, args=args, img_size=img_size)
    #     index = ['init']
    #     index.extend([f'{a}' for a in range(iter_cnt)])


    #     df = pd.concat([df, pd.Series(accuracy, index)])
    #     df.to_pickle(output_file+'.pkl')
    #     df.to_csv(output_file+'.csv')

    data_loader = data_module.val_dataloader(num_workers=16).__iter__()
    accuracy = evaluate(model, data_loader, args=args, img_size=img_size)
    df = pd.concat([df, pd.Series(accuracy)])
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

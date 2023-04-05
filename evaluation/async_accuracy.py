import argparse
import logging
import os
import numpy as np
import pandas as pd
import torch
import torchmetrics.functional as pl_metrics
import torch_geometric
import pytorch_lightning as pl

from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from tqdm import tqdm
tprint = tqdm.write
from typing import Iterable, Tuple
import aegnn
from aegnn.asyncronous.base.utils import causal_radius_graph


import signal


def signal_handler(signal, frame):
    global INT
    INT = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file", help="Path of model to evaluate.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--test-samples", default=None, type=int)

    parser.add_argument("--radius", default=3.0, help="radius of radius graph generation")
    parser.add_argument("--max-num-neighbors", default=32, help="max. number of neighbors in graph")
    parser.add_argument("--pooling-size", default=(12, 10))

    parser = aegnn.datasets.EventDataModule.add_argparse_args(parser)
    return parser.parse_args()


def sample_initial_data(sample, num_events: int, radius: float, edge_attr, max_num_neighbors: int):
    data = Data(x=sample.x[:num_events], pos=sample.pos[:num_events])
    subset = torch.arange(num_events)
    data.edge_index, data.edge_attr = torch_geometric.utils.subgraph(subset, sample.edge_index, sample.edge_attr)
    nxt_event_idx = num_events
    return data, nxt_event_idx

def sample_new_data(sample, nxt_event_idx):
    x_new = sample.x[nxt_event_idx, :].view(1, -1)
    pos_new = sample.pos[nxt_event_idx, :3].view(1, -1)  #TODO: :2 ? no time?
    event_new = Data(x=x_new, pos=pos_new, batch=torch.zeros(1, dtype=torch.long))
    nxt_event_idx += 1
    return event_new, nxt_event_idx

@torch.no_grad()
def calibre_quant(model_eval, data_loader, args):

    if isinstance(model_eval, pl.LightningModule):
        model = model_eval.model
        model.device =  model_eval.device
    elif isinstance(model, torch.nn.Module):
        model = model_eval
    else:
        raise TypeError(f'The type of model is {type(model)}, not a `torch.nn.Module` or a `pl.LightningModule`')

    from copy import deepcopy
    unfused_model = deepcopy(model)
    unfused_model = unfused_model.to(model_eval.device)
    unfused_model.eval()

    assert model.fused is False
    assert model.quantized is False
    model.to_fused()
    assert model.fused is True
    assert model.quantized is False

    model.eval()

    # calibration
    num_test_samples = 2460
    unfused_correct = 0
    fused_correct = 0
    for i, sample in enumerate(tqdm(data_loader, position=1, desc='Samples', total=num_test_samples)):
        torch.cuda.empty_cache()
        if i==num_test_samples: break
        # tprint(f"\nSample {i}, file_id {sample.file_id}:")

        sample = sample.to(model.device)
        tot_nodes = sample.num_nodes

        unfused_test_sample = sample.clone().detach().to(model.device)
        output_unfused = unfused_model.forward(unfused_test_sample)
        y_unfused = torch.argmax(output_unfused, dim=-1)
        # tprint(f'unfused output = {output_unfused}')
        unfused_hit = torch.allclose(y_unfused, unfused_test_sample.y)
        if unfused_hit: unfused_correct += 1

        fused_test_sample = sample.clone().detach().to(model.device)
        output_fused = model.forward(fused_test_sample)
        y_fused = torch.argmax(output_fused, dim=-1)
        # tprint(f'  fused output = {output_fused}')
        fused_hit = torch.allclose(y_fused, fused_test_sample.y)
        if fused_hit: fused_correct += 1

        # diff = torch.allclose(y_unfused, y_fused)
        # if diff is not True:
        #     print(i)
        #     print(f'unfused output = {output_unfused}')
        #     print(f'  fused output = {output_fused}')
    unfused_acc = unfused_correct / num_test_samples
    fused_acc = fused_correct / num_test_samples

    tprint(f'unfused_acc = {unfused_acc}')
    tprint(f'fused_acc = {fused_acc}')

    # quantization
    model.quant()
    assert model.quantized is True

    # quantization test
    quant_correct = 0
    for i, sample in enumerate(tqdm(data_loader, position=1, desc='Samples', total=num_test_samples)):
        torch.cuda.empty_cache()
        if i==num_test_samples: break
        # tprint(f"\nSample {i}, file_id {sample.file_id}:")

        sample = sample.to(model.device)
        tot_nodes = sample.num_nodes

        output_quant = model.forward(sample)
        y_quant = torch.argmax(output_quant, dim=-1)
        # tprint(f'  quant output = {output_quant}')
        quant_hit = torch.allclose(y_quant, sample.y)
        if quant_hit: quant_correct += 1
    quant_acc = quant_correct / num_test_samples
    tprint(f'quant_acc = {quant_acc}')

    return model_eval

@torch.no_grad()
def evaluate(model, data_loader, args, img_size, init_event: int = None, iter_cnt: int = None) -> float:
    predss = []
    targets = []

    edge_attr = torch_geometric.transforms.Cartesian(cat=False, max_value=args.radius)

    from copy import deepcopy
    sync_model = deepcopy(model.model)
    sync_model = sync_model.to(model.device)
    sync_model.eval()

    async_model = aegnn.asyncronous.make_model_asynchronous(model, args.radius, img_size, edge_attr)
    async_model.eval()

    df = pd.DataFrame()
    output_file = '/users/yyang22/thesis/aegnn_project/aegnn_results/mid_result'

    if args.test_samples is not None:
        num_test_samples = args.test_samples
        tprint(f"Will test {num_test_samples} sample(s) in the dataset")
    else:
        num_test_samples = len(data_loader)
        tprint(f"Will test all samples in the dataset")

    max_nodes = 0
    # for NCars dataset, #events: min=500 at 1422, max=40810 at 219, mean=3920; #samples = 2462
    for i, sample in enumerate(tqdm(data_loader, position=1, desc='Samples', total=num_test_samples)):
        torch.cuda.empty_cache()
        if i==num_test_samples: break
        tprint(f"\nSample {i}, file_id {sample.file_id}:")

        sample = sample.to(model.device)
        tot_nodes = sample.num_nodes
        if tot_nodes > max_nodes: max_nodes = tot_nodes

        sync_test_sample = sample.clone().detach()
        output_sync = sync_model.forward(sync_test_sample)
        y_sync = torch.argmax(output_sync, dim=-1)
        tprint(f' sync output = {output_sync}')
        # sample.y = y_sync  # Debug only: for random sample input
        targets.append(sample.y)

        async_model = aegnn.asyncronous.reset_async_module(async_model)
        aegnn.asyncronous.register_sync_graph(async_model, sample) #TODO: for debug


        init_num_event = 2
        # init_num_event = tot_nodes - 0

        sub_predss = []

        events_initial, nxt_event_idx = sample_initial_data(sample, init_num_event, args.radius, edge_attr, args.max_num_neighbors)
        while nxt_event_idx < tot_nodes:
            if events_initial.edge_index.numel() > 0:
                break
            else:
                sub_predss.append(torch.tensor([0.], device=model.device))
                events_initial, nxt_event_idx = sample_initial_data(sample, nxt_event_idx+1, args.radius, edge_attr, args.max_num_neighbors)
        tprint(f'1st edge starts from node {nxt_event_idx}; former predictions default to 0.0')

        # init stage
        output_new = async_model.forward(events_initial)
        y_init = torch.argmax(output_new, dim=-1)
        sub_predss.append(y_init)

        # iterative adding nodes stage
        with tqdm(total=(tot_nodes-nxt_event_idx), position=0, leave=False, desc='Events') as pbar:
            while nxt_event_idx < tot_nodes:
                torch.cuda.empty_cache()

                event_new, nxt_event_idx = sample_new_data(sample, nxt_event_idx)
                event_new = event_new.to(model.device)

                output_new = async_model.forward(event_new)
                y_new = torch.argmax(output_new, dim=-1)

                sub_predss.append(y_new)
                pbar.update(1)
                if INT: break
        tprint(f'async output = {output_new}')

        # Test if graphs are the same
        # aegnn_graph = async_model.model.conv1.asy_graph
        # aegnn_graph.edge_index = torch_geometric.utils.to_undirected(aegnn_graph.edge_index)
        # ordered_sync_edge = torch_geometric.utils.to_undirected(sync_test_sample.edge_index)
        # tprint(f'asy graph == sync graph ? {torch.allclose(aegnn_graph.edge_index, ordered_sync_edge)}')


        sub_preds = torch.cat(sub_predss)

        column_name = pd.MultiIndex.from_tuples([(i, sample.y.cpu().item())], names=['i', 'gnd_truth'])
        df.insert(len(targets)-1, column_name, pd.Series(sub_preds.cpu()), allow_duplicates=True)
        df.to_pickle(output_file+'.pkl')
        df.to_csv(output_file+'.csv')

        predss.append(sub_preds)
        if INT: break



    # "torch.nested" is not yet supported!
    # preds_nt = torch.nested.nested_tensor(predss)
    # preds = torch.nested.to_padded_tensor(preds_nt, 0.0)


    max_nodes += 100
    for i, sub_preds in enumerate(predss):
        tmp = torch.nn.functional.pad(sub_preds, (0,max_nodes-len(sub_preds)), value=sub_preds[-1].item())
        predss[i] = tmp.unsqueeze(0)

    preds = torch.cat(predss, dim=0)
    target = torch.cat(targets).unsqueeze(1)
    tot_accuracies = []
    for j in tqdm(range(max_nodes), position=2, desc='Acc Calc', leave=False):
        tot_accuracy = pl_metrics.accuracy(preds=preds[:,j], target=target)
        tot_accuracy = tot_accuracy.item()
        tot_accuracies.append(tot_accuracy)
    return tot_accuracies


def main(args, model, data_module):

    img_size = list(data_module.dims)

    df = pd.DataFrame()
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "aegnn_results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "async_accuracy")

    # data_loader = data_module.val_dataloader(num_workers=16).__iter__()
    data_loader = data_module.val_dataloader(num_workers=16)

    model = calibre_quant(model, data_loader, args)
    accuracy = evaluate(model, data_loader, args=args, img_size=img_size)
    df = pd.concat([df, pd.Series(accuracy)])
    df.to_pickle(output_file+'.pkl')
    df.to_csv(output_file+'.csv')
    tprint(f"Results are logged in {output_file}.*")
    return df


if __name__ == '__main__':
    pl.seed_everything(12345)
    args = parse_args()
    if args.debug:
        _ = aegnn.utils.loggers.LoggingLogger(None, name="debug")

    torch.set_printoptions(precision=16)

    model_eval = torch.load(args.model_file).to(args.device)
    model_eval.eval()
    dm = aegnn.datasets.by_name(args.dataset).from_argparse_args(args)
    dm.setup()

    signal.signal(signal.SIGINT, signal_handler)
    INT = False

    main(args, model_eval, dm)

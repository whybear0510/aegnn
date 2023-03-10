import aegnn
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
from aegnn.asyncronous.base.utils import causal_radius_graph

from time import time
import signal
import sys

def signal_handler(signal, frame):
    global INT
    INT = True


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
def evaluate(model, data_loader: Iterable[Batch], args, img_size, init_event: int = None, iter_cnt: int = None) -> float:
    predss = []
    targets = []

    # TODO: for debug
    # for layer in model.model.children():
    #     # if layer.__class__.__name__ in torch_geometric.nn.conv.__all__:
    #     if hasattr(layer, 'reset_parameters'):
    #         layer.reset_parameters()
    # model.eval()



    edge_attr = torch_geometric.transforms.Cartesian(cat=False, max_value=3.0)

    from copy import deepcopy
    sync_model = deepcopy(model.model)
    sync_model = sync_model.to(model.device)
    sync_model.eval()

    async_model = aegnn.asyncronous.make_model_asynchronous(model, args.radius, img_size, edge_attr)
    async_model.eval()

    # w1 = model.model.conv1.weight
    # w2 = model.model.conv2.weight
    # w3 = model.model.conv3.weight
    # w4 = model.model.conv4.weight
    # w5 = model.model.conv5.weight
    # w6 = model.model.conv6.weight
    # w7 = model.model.conv7.weight
    # wf = model.model.fc.weight

    def load_test_sample(n, args):
        import numpy as np
        def load_ncal_data(raw_file: str):
            f = open(raw_file, 'rb')
            raw_data = np.fromfile(f, dtype=np.uint8)
            f.close()

            raw_data = np.uint32(raw_data)
            all_y = raw_data[1::5] //3 + 1.0
            all_x = raw_data[0::5] //3 + 1.0
            all_p = (raw_data[2::5] & 128) >> 7  # bit 7
            all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
            all_ts = all_ts / 1e6  # µs -> s
            all_p = all_p.astype(np.float32)
            # all_p[all_p == 0] = -1
            events = np.column_stack((all_x, all_y, all_ts, all_p)).astype(np.float32)
            events = torch.from_numpy(events[:6000, :])
            events_data = torch_geometric.data.Data(x=events[:, 3].view(-1,1), pos=events[:,:3])
            return events_data

        # path = "/space/yyang22/datasets/data/storage/ncaltech101/training/buddha/image_0004.bin"
        s = "%04d" % (n+1)
        path = "/space/yyang22/datasets/data/storage/ncaltech101/training/buddha/image_"+s+".bin"
        events_graph = load_ncal_data(path)
        tprint('Generating graph...')
        events_graph.edge_index = causal_radius_graph(events_graph.pos, r=args.radius, max_num_neighbors=args.max_num_neighbors).to(model.device)
        tprint('Done')
        return events_graph

    def load_ncars(n, args):
        # path = "/space/yyang22/datasets/data/storage/ncars/validation/sequence_12961/"
        # events = np.loadtxt(path+'events.txt')
        # is_car = np.loadtxt(path+'is_car.txt')
        # events = events.astype(np.float32)
        # is_car = is_car.astype(np.float32)


        # is_car = torch.from_numpy(is_car).to(model.device)

        # events_p = torch.from_numpy(events[:,3]).to(model.device).view(-1,1)
        # events_pos = torch.from_numpy(events[:,:3]).to(model.device)
        # events_pos[:,2] *= 5e-5

        # events_graph = Data(x=events_p, pos=events_pos, y=is_car)

        # tprint('Generating graph...')
        # events_graph.edge_index = causal_radius_graph(events_graph.pos, r=args.radius, max_num_neighbors=args.max_num_neighbors).to(model.device)
        # tprint('Done')

        processed_path = "/space/yyang22/datasets/data/storage/ncars/processed/validation/sequence_14660"
        events_graph = torch.load(processed_path).to(model.device)

        print(events_graph)
        return events_graph




    df = pd.DataFrame()
    output_file = '/users/yyang22/thesis/aegnn_project/aegnn_results/mid_result'

    num_test_samples = 40
    max_nodes = 0

    # for NCars dataset, #events: min=500 at 1422, max=40810 at 219, mean=3920; #samples = 2462
    for i, batch in enumerate(tqdm(data_loader, position=1, desc='Samples', total=num_test_samples)):

        if i==num_test_samples: break

        # if i!=43:continue  #!debug: at 43, the sample init func will be fail at (index<size[i]) (??)


        sample = batch
        sample = sample.to(model.device)
        # old_attr = sample.edge_attr
        # sample = edge_attr(sample)
        # new_attr = sample.edge_attr
        # diff =  torch.allclose(old_attr,new_attr)
        # print(diff)
        tot_nodes = sample.num_nodes
        print(sample.file_id)
        if tot_nodes > max_nodes: max_nodes = tot_nodes

        # #TODO: debug mode: random graph
        # del sample
        # # x_n = torch.round(torch.rand(tot_nodes+1, device=model.device), decimals=0).view(-1,1)
        # # pos_n = torch.rand(tot_nodes+1,3)*98.0
        # # pos_n[:,:2] = torch.round(pos_n[:,:2])
        # # pos_n[:,2] *= 1e-10
        # # pos_n = pos_n[torch.argsort(pos_n[:,2]).squeeze(),:]
        # # pos_n = pos_n.to(model.device)
        # # random_sample = Data(x = x_n[:tot_nodes,:],pos = pos_n[:tot_nodes,:])
        # tot_nodes = 30
        # random_sample = Data(
        #     x = torch.round(torch.rand(tot_nodes, device=model.device), decimals=0).view(-1,1),
        #     pos = torch.tensor([[0,0,0e-10],[0,2,2e-10],[0,4,4e-10],[0,6,6e-10],[0,8,8e-10],
        #                         [0,10,10e-10],[0,12,12e-10],[0,14,14e-10],[0,16,16e-10],[0,18,18e-10],
        #                         [0,20,20e-10],[0,22,22e-10],[0,24,24e-10],[0,26,26e-10],[0,28,28e-10],
        #                         [0,30,30e-10],[0,32,32e-10],[0,34,34e-10],[0,36,36e-10],[0,38,38e-10],
        #                         [0,40,40e-10],[0,42,42e-10],[0,44,44e-10],[0,46,46e-10],[0,48,48e-10],
        #                         [0,50,50e-10],[0,52,52e-10],[0,54,54e-10],[0,56,56e-10],[0,58,58e-10]],
        #                        device=model.device, dtype=torch.float)
        # )
        # from aegnn.asyncronous.base.utils import causal_radius_graph
        # print('Generating random graph...')
        # random_sample.edge_index = causal_radius_graph(random_sample.pos, r=args.radius, max_num_neighbors=args.max_num_neighbors).to(model.device)
        # random_sample = edge_attr(random_sample)
        # sample = random_sample
        # test_sample = torch.load("/space/yyang22/datasets/data/storage/ncars/processed/test_sample")

        # # #TODO: debug mode: graph from ncaltech
        # del sample
        # del tot_nodes
        # test_sample = load_test_sample(i, args)
        # sample = test_sample.to(model.device)
        # tot_nodes = sample.num_nodes
        # sample = edge_attr(sample)

        #TODO: debug mode: graph from ncars
        # del sample
        # del tot_nodes
        # test_sample = load_ncars(i, args)
        # sample = test_sample.to(model.device)
        # tot_nodes = sample.num_nodes
        # sample = edge_attr(sample)


        # model_t = torch.load(args.model_file).to(args.device)
        # input_shape = torch.tensor([*dm.dims, sample.pos.shape[-1]], device=model.device)
        # model_t = aegnn.models.networks.GraphRes(dm.name, input_shape, dm.num_classes, pooling_size=args.pooling_size)
        # model_t.to(model.device)
        # model_t.conv1.weight = w1
        # model_t.conv2.weight = w2
        # model_t.conv3.weight = w3
        # model_t.conv4.weight = w4
        # model_t.conv5.weight = w5
        # model_t.conv6.weight = w6
        # model_t.conv7.weight = w7
        # model_t.fc.weight = wf

        # model_t.norm1 = model.model.norm1
        # model_t.norm2 = model.model.norm2
        # model_t.norm3 = model.model.norm3
        # model_t.norm4 = model.model.norm4
        # model_t.norm5 = model.model.norm5
        # model_t.norm6 = model.model.norm6
        # model_t.norm7 = model.model.norm7

        # model_t.eval()
        sync_test_sample = sample.clone().detach()
        # sync_test_sample = load_ncars(i, args)
        # sync_test_sample = sync_test_sample.to(model.device)
        # sync_test_sample = edge_attr(sync_test_sample)
        output_sync = sync_model.forward(sync_test_sample)
        y_sync = torch.argmax(output_sync, dim=-1)
        tprint(f'\nsync output = {output_sync}')
        # sample.y = y_sync
        # targets.append(y_sync) #TODO: for debug: only for random sample input
        targets.append(sample.y)

        # async_model = aegnn.asyncronous.make_model_asynchronous(model_t, args.radius, img_size, edge_attr)
        # async_model.eval()


        async_model = aegnn.asyncronous.reset_async_module(async_model)
        aegnn.asyncronous.register_sync_graph(async_model, sample) #TODO: for debug

        if init_event is None and iter_cnt is not None:
            init_num_event = tot_nodes - iter_cnt
        elif init_event is not None:
            init_num_event = init_event
        else:
            init_num_event = 2  # guarantee to have an edge between nodes #!debug from here: 54: reason: after torch.unique, 'size' is lost
            # init_num_event = tot_nodes - 1

        sub_predss = []


        # events_initial, nxt_event_idx = sample_initial_data(sample, init_num_event, args.radius, edge_attr, args.max_num_neighbors)
        # events_initial = events_initial.to(model.device)
        # if events_initial.edge_index.nelement() == 0:
        #     continue

        events_initial, nxt_event_idx = sample_initial_data(sample, init_num_event, args.radius, edge_attr, args.max_num_neighbors)
        while nxt_event_idx < tot_nodes:
            if events_initial.edge_index.numel() > 0:
                break
            else:
                events_initial, nxt_event_idx = sample_initial_data(sample, nxt_event_idx+1, args.radius, edge_attr, args.max_num_neighbors)
        tprint(f'1st edge started from node {nxt_event_idx}')
        output_new = async_model.forward(events_initial)
        y_init = torch.argmax(output_new, dim=-1)
        sub_predss.append(y_init)

        # print(f'timestamp = {events_initial.pos[-1,2].item()}')

        # code_time_diff = []
        # idx=0
        # while idx < len(code_time)-1:
        #     code_time_diff.append(code_time[idx+1]-code_time[idx])
        #     idx += 1
        # print(f'init:{code_time_diff}')


        with tqdm(total=(tot_nodes-nxt_event_idx), position=0, leave=False, desc='Events') as pbar:
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
                if INT: break

        tprint(f'async output = {output_new}')
        # exit()

        # print(f'sample = {sample}')
        # print(f'async = {async_model.model.conv1.asy_graph}')
        # exit()

        # async_graph = model.model.conv1.asy_graph
        # sync_graph = model.model.conv1.sync_graph
        # a,index = torch.sort(async_graph.edge_index[0, :])
        # # a1,_ = torch.sort(sync_graph.edge_index[0, :])
        # # print(torch.allclose(async_graph.edge_index,sync_graph.edge_index))
        # a1 = async_graph.edge_index[:,index]
        # print(torch.allclose(sync_graph.edge_index,a1))
        # aegnn_graph = async_model.model.conv1.asy_graph
        # aegnn_graph.edge_index = torch_geometric.utils.to_undirected(aegnn_graph.edge_index)
        # print(f'asy graph == sync graph ? {torch.allclose(aegnn_graph.edge_index, sync_test_sample.edge_index)}')
        # ordered_sync_edge = torch_geometric.utils.to_undirected(sync_test_sample.edge_index)
        # print(f'asy graph == sync graph ? {torch.allclose(aegnn_graph.edge_index, ordered_sync_edge)}')


        sub_preds = torch.cat(sub_predss)

        # df = pd.concat([df, pd.Series(sub_preds.cpu())], axis=1)
        # df.columns = [truth.cpu() for truth in targets]
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
    # data_loader = data_module.train_dataloader().__iter__()
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

    model_eval = torch.load(args.model_file).to(args.device)
    model_eval.eval()
    dm = aegnn.datasets.by_name(args.dataset).from_argparse_args(args)
    dm.setup()

    signal.signal(signal.SIGINT, signal_handler)
    INT = False

    main(args, model_eval, dm)

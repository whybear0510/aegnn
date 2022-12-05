import aegnn
import argparse
import itertools
import logging
import os
import pandas as pd
import torch
import torch_geometric

from torch_geometric.data import Data
from tqdm import tqdm
from typing import List


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius", default=3.0, help="radius of radius graph generation")
    parser.add_argument("--max-num-neighbors", default=32, help="max. number of neighbors in graph")
    parser.add_argument("--pooling-size", default=(10, 10))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--num-trials", default=400, type=int, help="run times")

    return parser.parse_args()


##################################################################################################
# Graph Generation ###############################################################################
##################################################################################################
def sample_initial_data(sample, num_events: int, radius: float, edge_attr, max_num_neighbors: int):
    data = Data(x=sample.x[:num_events], pos=sample.pos[:num_events])
    data.batch = torch.zeros(data.num_nodes, device=data.x.device)
    data.edge_index = torch_geometric.nn.radius_graph(data.pos, r=radius, max_num_neighbors=max_num_neighbors).long()
    data.edge_attr = edge_attr(data).edge_attr

    edge_counts_avg = data.edge_index.shape[1] / num_events
    logging.debug(f"Average edge counts in initial data = {edge_counts_avg}")
    return data


def create_and_run_model(dm, num_events: int, index: int, device: torch.device, args: argparse.Namespace, **kwargs):
    edge_attr = torch_geometric.transforms.Cartesian(cat=False, max_value=10.0)
    dataset = dm.train_dataset
    assert dm.shuffle is False  # ensuring same samples over experiments

    # Sample initial data of certain length from dataset sample. Sample num_events samples from one
    # dataset, and create the subsequent event as the one to be added.
    sample = dataset[index % len(dataset)] # len(dataset) = 12961
    sample.pos = sample.pos[:, :2]
    num_nodes = min(sample.num_nodes, num_events)
    events_initial = sample_initial_data(sample, num_events, args.radius, edge_attr, args.max_num_neighbors)
    # max num_nodes in Ncars dataset is 10000, min is 500

    index_new = min(num_events, sample.num_nodes - 1)
    x_new = sample.x[index_new, :].view(1, -1)
    pos_new = sample.pos[index_new, :2].view(1, -1)
    event_new = Data(x=x_new, pos=pos_new, batch=torch.zeros(1, dtype=torch.long))

    # Initialize model and make it asynchronous (recognition model, so num_outputs = num_classes of input dataset).
    input_shape = torch.tensor([*dm.dims, events_initial.pos.shape[-1]], device=device)
    model = aegnn.models.networks.GraphRes(dm.name, input_shape, dm.num_classes, pooling_size=args.pooling_size)
    model.to(device)
    model = aegnn.asyncronous.make_model_asynchronous(model, args.radius, list(dm.dims), edge_attr, **kwargs)

    # Run experiment, i.e. initialize the asynchronous graph and iteratively add events to it.
    _ = model.forward(events_initial.to(device))  # initialization
    _ = model.forward(event_new.to(device))
    del events_initial, event_new
    return model,num_nodes


##################################################################################################
# Logging ########################################################################################
##################################################################################################
def get_log_values(model, attr: str, log_key: str, **log_dict):
    """"Get log values for the given attribute key, both for each layer and in total and for the dense and sparse
    update. Thereby, approximate the logging for the dense with the data for the initial update as
    count(initial events) >>> count(new events)
    """
    assert hasattr(model, attr)
    log_values = []
    for layer, nn in model._modules.items():
        if hasattr(nn, attr):
            logs = getattr(nn, attr)
            log_values.append({"layer": layer, log_key: logs[0], "model": "gnn_dense", **log_dict})
            for log_i in logs[1:]:
                log_values.append({"layer": layer, log_key: log_i, "model": "ours", **log_dict})

    logs = getattr(model, attr)
    log_values.append({"layer": "total", log_key: logs[0], "model": "gnn_dense", **log_dict})
    for log_i in logs[1:]:
        log_values.append({"layer": "total", log_key: log_i, "model": "ours", **log_dict})

    total_dense = logs[0]
    total_sparse = log_i

    return log_values, total_dense, total_sparse


##################################################################################################
# Experiments ####################################################################################
##################################################################################################
def run_experiments(dm, args, experiments: List[int], num_trials: int, device: torch.device, **model_kwargs
                    ) -> pd.DataFrame:

    output_file_flops = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "aegnn_results", "flops")
    os.makedirs(os.path.dirname(output_file_flops), exist_ok=True)
    results_df_flops = pd.DataFrame()

    flops_track = dict()
    flops_track['dense'] = []
    flops_track['sparse'] = []

    output_file_runtime = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "aegnn_results", "runtime")
    os.makedirs(os.path.dirname(output_file_runtime), exist_ok=True)
    results_df_runtime = pd.DataFrame()

    runs = list(itertools.product(experiments, list(range(num_trials))))
    #[(25000,0),(25000,1),...,(25000,99)]
    for num_events, exp_id in tqdm(runs):
        model, num_nodes = create_and_run_model(dm, num_events, index=exp_id, args=args, device=device, **model_kwargs)

        # Get the logged flops and timings, both layer-wise and in total.
        results_flops, flops_dense, flops_sparse = get_log_values(model, attr="asy_flops_log", log_key="flops", num_events=num_nodes, sample=exp_id)
        results_df_flops = pd.concat([results_df_flops, pd.DataFrame(results_flops)], ignore_index=True)

        flops_per_events_dense = flops_dense / num_nodes
        flops_per_events_sparse = flops_sparse / num_nodes

        flops_track['dense'].append(flops_per_events_dense)
        flops_track['sparse'].append(flops_per_events_sparse)


        results_runtime, runtime_dense, runtime_sparse = get_log_values(model, attr="asy_runtime_log", log_key="runtime", num_events=num_nodes, sample=exp_id)
        results_df_runtime = pd.concat([results_df_runtime, pd.DataFrame(results_runtime)], ignore_index=True)


        # Fully reset run to ensure independence between subsequent experiments.
        del model  # fully delete model
        torch.cuda.empty_cache()  # clear memory

    dm_flops_per_events_dense = {"layer": "avg", "flops": sum(flops_track['dense'])/len(flops_track['dense']), "model": "gnn_dense", "sample": len(flops_track['dense'])}
    dm_flops_per_events_sparse = {"layer": "avg", "flops": sum(flops_track['sparse'])/len(flops_track['sparse']), "model": "gnn_sparse", "sample": len(flops_track['sparse'])}
    results_df_flops = pd.concat([results_df_flops, pd.DataFrame(dm_flops_per_events_dense, index=[0]), pd.DataFrame(dm_flops_per_events_sparse, index=[1])], ignore_index=True)
    results_df_flops = results_df_flops.astype({'num_events':'Int64', 'sample':'Int64'})

    results_df_flops.to_pickle(output_file_flops+'.pkl')
    results_df_flops.to_csv(output_file_flops+'.csv')
    print(f"Results are logged in {output_file_flops}.*")



    results_df_runtime = results_df_runtime.astype({'num_events':'Int64', 'sample':'Int64'})
    results_df_runtime.to_pickle(output_file_runtime+'.pkl')
    results_df_runtime.to_csv(output_file_runtime+'.csv')

    print(f"Results are logged in {output_file_runtime}.*")
    return results_df_flops, results_df_runtime


if __name__ == '__main__':
    arguments = parse_args()
    if arguments.debug:
        _ = aegnn.utils.loggers.LoggingLogger(None, name="debug")

    data_module = aegnn.datasets.NCars(batch_size=1, shuffle=False)
    data_module.setup()
    # event_counts = [25000]
    event_counts = [50000]
    # event_counts = list(np.linspace(1000, 15000, num=10).astype(int))
    run_experiments(data_module, arguments, experiments=event_counts, num_trials=arguments.num_trials,
                    device=torch.device(arguments.device), log_flops=True, log_runtime=True)

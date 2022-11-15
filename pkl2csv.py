import pickle as pkl
import pandas as pd
import aegnn
import torch
import torch_geometric

from torch_geometric.data import Data
from tqdm import tqdm
from typing import List

data_module = aegnn.datasets.NCars(batch_size=1, shuffle=False)
data_module.setup()
dataset = data_module.train_dataset
num_trials=4000
nodes=[]
for index in tqdm(range(num_trials)):
    sample = dataset[index % len(dataset)]
    nodes.append(sample.num_nodes)

# print(nodes)



with open("../aegnn_results/flops.pkl", "rb") as f:
    object = pkl.load(f)
    
df = pd.DataFrame(object)
df.to_csv(r'../aegnn_results/flops.csv')

ls = df.values.tolist()

Mflops=[]
acc_Mflop = 0
acc_Mflop_per_ev = 0
for i in range(num_trials): 
    Mflop = (ls[(33+68*i)][1])/1000000
    Mflops.append(Mflop)
    acc_Mflop += (Mflop)
    acc_Mflop_per_ev += Mflop/nodes[i]

avg_Mflop =  acc_Mflop/num_trials
avg_Mflop_per_ev = acc_Mflop_per_ev/num_trials
# print(flops)
print(f'acc_Mflop_per_ev = {acc_Mflop_per_ev}, avg_Mflop_per_ev = {avg_Mflop_per_ev}')

stdMflops=[]
stdacc_Mflop = 0
stdacc_Mflop_per_ev = 0
for i in range(num_trials): 
    stdMflop = (ls[(32+68*i)][1])/1000000
    stdMflops.append(stdMflop)
    stdacc_Mflop += (stdMflop)
    stdacc_Mflop_per_ev += stdMflop/nodes[i]

stdavg_Mflop =  stdacc_Mflop/num_trials
stdavg_Mflop_per_ev = stdacc_Mflop_per_ev/num_trials
# print(flops)
print(f'dense: std acc_Mflop_per_ev = {stdacc_Mflop_per_ev}, std avg_Mflop_per_ev = {stdavg_Mflop_per_ev}')



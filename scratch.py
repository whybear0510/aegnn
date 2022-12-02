import torch
import torch_geometric
import pytorch_lightning as pl
# import pytorch_lightning.metrics.functional as pl_metrics
import torchmetrics.functional as pl_metrics
import numpy as np

from torch.nn.functional import softmax
from typing import Tuple
from tqdm import tqdm
import aegnn

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'





batches = [1,64]
compare_preds=[]
compare_y=[]
compare_load=[]
ass=[]
debugs=[]
multiple = int(batches[-1]/batches[0])

for j, batch_size in enumerate(batches):
    predss=[]
    yss=[]
    def validation_step(self, batch: torch_geometric.data.Batch, batch_idx: int) -> torch.Tensor:
        
        self.eval()
        # outputs,debug = self.forward(data=batch)
        outputs = self.forward(data=batch)
        debug=None
        y_prediction = torch.argmax(outputs, dim=-1)

        acc = pl_metrics.accuracy(preds=y_prediction, target=batch.y)

        # print(y_prediction)
        # print(batch.y)
        # print(torch.abs(y_prediction-batch.y))
        predss.append(y_prediction)
        yss.append(batch.y)

        return acc, debug


    model_file = '/users/yyang22/thesis/aegnn_project/aegnn_results/training_results/checkpoints/ncars/recognition/20221125084923/epoch=99-step=20299.pt'
    self = torch.load(model_file).to(torch.device('cuda'))
    # dm = aegnn.datasets.by_name('ncars')(batch_size = batch_size, num_workers=0, shuffle=False)
    dm = aegnn.datasets.by_name('ncars')(batch_size = 1, num_workers=0, shuffle=False)
    dm.setup()
    dm_acc = []
    times = 128//(batch_size)-1
    

    # for i,batch in tqdm(enumerate(dm.val_dataloader())):
    #     batch = batch.to(self.device)
    #     ass.append(batch.x) 
    #     dm_acc.append(validation_step(self, batch, i)) 
    #     if i==(times): break

    for i,sample in enumerate(tqdm(dm.val_dataloader())):
        if i==(1): break
    batch = torch_geometric.data.Batch.from_data_list([sample for x in range(batch_size)])
    # compare_load.append(batch.x.detach().cpu().numpy())
    batch = batch.to(self.device)
    val_acc, debug = validation_step(self, batch, i)
    debugs.append(debug)


# load_1 = np.concatenate([compare_load[0] for i in range(batches[-1])], axis=0)
# load_bs = compare_load[1]
# print((load_1==load_bs).all())
# print('load')

debug_1 = np.concatenate([debugs[0] for i in range(multiple)], axis=0)
debug_bs = debugs[1]
diff = debug_1 - debug_bs
print(np.where(debug_1!=debug_bs))
print(np.sum(np.abs(diff)))
print('debug')



    # tensor_preds = torch.cat(predss, dim=-1)
    # tensor_ys = torch.cat(yss, dim=-1)


#     acc_tensor = torch.tensor(dm_acc, dtype=torch.double)
#     tot_acc = torch.mean(acc_tensor)
#     print(tot_acc.item())
#     compare_preds.append(tensor_preds)
#     compare_y.append(tensor_ys)


# tensor_ass_16 = torch.cat(ass[:8], dim=0)
# tensor_ass_128 = ass[8]

# diff = tensor_ass_16 - tensor_ass_128
# print(tensor_ass_16 - tensor_ass_128)
# print(torch.sum(torch.abs(diff)))

# diff_preds = compare_preds[3] - compare_preds[4]
# diff_y = compare_y[3] - compare_y[4]
# print(f'preds:{diff_preds}')
# print(f'ys={diff_y}')


import os

def cat_cmd(*cmd_list, mode='pipe'):
    modes = {
        'pipe': ' && ',
        'seq': ' ; ',
        'until_success': ' || '
    }
    final_cmd = ''

    print(f'\nWill execute following cmds in mode {mode}:\n')
    for cmd in cmd_list:
        print(f'{cmd}')
        print_cmd = f'echo ;echo Running: ;echo {cmd} ;echo ;'
        final_cmd += (print_cmd + cmd + modes[mode])
    final_cmd += ':'  # empty cmd
    print(f'\n')
    return final_cmd




model = 'graph_res' # try graph_wen?
task = 'recognition'
dataset = 'ncars'
which_gpu = 2
dim = 3
device = 'cuda'

max_epochs = 100
batch_size = 64
init_lr = 0.001
w_decay = 0.00
act = 'relu'
run_name = '--run-name PointNetConv'
grid_div = 8
# run_name = None

# sweep_config = {
#     'method': 'random',
#     'name': 'sweep',
#     'metric': {'goal': 'maximize', 'name': "Val/Accuracy"},
#     'parameters':
#     {
#         'max_epochs': {'values': [30,50,100]},
#         'learning_rate': {'max': 0.01, 'min': 0.0005},
#         'batch_size': {'values': [16, 32, 64, 128]},
#         'weight_decay': {'max': 0.01, 'min': 0.001}
#      }
# }

# eg: CUDA_VISIBLE_DEVICES=1 python3 ../scripts/train.py graph_res --task recognition --dataset ncars --batch-size 64 --dim 3 --init-lr 0.001 --weight-decay 0.0 --act relu
cmd_train = f'CUDA_VISIBLE_DEVICES={which_gpu} python3 ../scripts/train.py {model} --task {task} --dataset {dataset} --batch-size {batch_size} --dim {dim} --max-epochs {max_epochs} --init-lr {init_lr} --weight-decay {w_decay} --act {act} --grid-div {grid_div} {run_name}'
cmd_cpresult = f'python get_latest_results.py'
# eg: CUDA_VISIBLE_DEVICES=1 python ../evaluation/accuracy_per_events.py /users/yyang22/thesis/aegnn_project/aegnn_results/training_results/latest/latest_model.pt --device cuda --dataset ncars --batch-size 64 && python pkl2csv.py
# eg: CUDA_VISIBLE_DEVICES=1 python ../evaluation/accuracy_per_events.py /users/yyang22/thesis/aegnn_project/aegnn_results/training_results/latest/latest_model.pt --device cuda --dataset ncars --batch-size 64 --fast-test
cmd_accuracy = f'CUDA_VISIBLE_DEVICES={which_gpu} python ../evaluation/accuracy_per_events.py /users/yyang22/thesis/aegnn_project/aegnn_results/training_results/latest/latest_model.pt --device {device} --dataset {dataset} --batch-size {batch_size}'


cmd = cat_cmd(cmd_train, cmd_cpresult, cmd_accuracy)
os.system(cmd)
# os.system(cmd_train)


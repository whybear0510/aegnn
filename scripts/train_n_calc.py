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
which_gpu = 6
dim = 3
device = 'cuda'

max_epochs = 100
batch_size = 64
init_lr = 0.001
w_decay = 0.00
act = 'relu'
run_name = '--run-name gcn_bug_fixed'
grid_div = 8
conv_type = 'gcn'
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

# eg: CUDA_VISIBLE_DEVICES=2 python3 ../scripts/train.py graph_res --task recognition --dataset ncars --batch-size 64 --dim 3 --init-lr 0.001 --weight-decay 0.0 --act relu --conv-type fuse --run-name teacher_fuse16_16_maxp
# eg: CUDA_VISIBLE_DEVICES=2 python3 ../scripts/train.py graph_res --task recognition --dataset ncars --batch-size 64 --dim 3 --init-lr 0.001 --weight-decay 0.0 --act relu --conv-type fuse --run-name fuse16_16_maxp --distill --teacher-model-path /users/yyang22/thesis/aegnn_project/aegnn_results/training_results/checkpoints/ncars/recognition/20230420224023/epoch=99-step=20299.pt --distill-t 2 --distill-alpha 0.95

# eg: CUDA_VISIBLE_DEVICES=2 python3 ../scripts/train.py graph_res --task recognition --dataset ncars --batch-size 64 --dim 3 --init-lr 0.001 --weight-decay 0.0 --act relu --grid-div 8 --conv-type pointnet_single --auto-lr-find --run-name cycliclr_auto_find
# eg: CUDA_VISIBLE_DEVICES=6 python3 ../scripts/train.py graph_res --task recognition --dataset ncars --batch-size 64 --dim 3 --init-lr 0.001 --weight-decay 0.0 --act relu --grid-div 8 --conv-type sage --run-name sage
cmd_train = f'CUDA_VISIBLE_DEVICES={which_gpu} python3 ../scripts/train.py {model} --task {task} --dataset {dataset} --batch-size {batch_size} --dim {dim} --max-epochs {max_epochs} --init-lr {init_lr} --weight-decay {w_decay} --act {act} --grid-div {grid_div} --conv-type {conv_type} {run_name}'
cmd_cpresult = f'python get_latest_results.py'
# eg: CUDA_VISIBLE_DEVICES=1 python ../evaluation/accuracy_per_events.py /users/yyang22/thesis/aegnn_project/aegnn_results/training_results/latest/latest_model.pt --device cuda --dataset ncars --batch-size 64 && python pkl2csv.py
# eg: CUDA_VISIBLE_DEVICES=1 python ../evaluation/accuracy_per_events.py /users/yyang22/thesis/aegnn_project/aegnn_results/training_results/latest/latest_model.pt --device cuda --dataset ncars --batch-size 64 --fast-test
cmd_accuracy = f'CUDA_VISIBLE_DEVICES={which_gpu} python ../evaluation/accuracy_per_events.py /users/yyang22/thesis/aegnn_project/aegnn_results/training_results/latest/latest_model.pt --device {device} --dataset {dataset} --batch-size {batch_size} --fast-test'

# eg: CUDA_VISIBLE_DEVICES=7 python3 ../evaluation/async_accuracy.py /users/yyang22/thesis/aegnn_project/aegnn_results/training_results/checkpoints/ncars/recognition/20230409231150/epoch=99-step=20299.pt --device cuda --dataset ncars --batch-size 1 --test-samples 600

# eg: CUDA_VISIBLE_DEVICES=3 python3 ../scripts/preprocessing.py --dataset ncars --num-workers 8

cmd = cat_cmd(cmd_train, cmd_cpresult, cmd_accuracy)
os.system(cmd)
# os.system(cmd_train)


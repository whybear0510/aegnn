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
which_gpu = 5
dim = 3
device = 'cuda'

max_epochs = 100
batch_size = 64
init_lr = 0.001
w_decay = 0.005



cmd_train = f'python3 ../scripts/train.py {model} --task {task} --dataset {dataset} --gpu {which_gpu} --batch-size {batch_size} --dim {dim} --max-epochs {max_epochs} --init-lr {init_lr} --weight-decay {w_decay}'
cmd_cpresult = f'python get_latest_results.py'
cmd_accuracy = f'python ../evaluation/accuracy_per_events.py /users/yyang22/thesis/aegnn_project/aegnn_results/training_results/latest/latest_model.pt --device {device} --dataset {dataset}'
cmd_csv = f'python pkl2csv.py'

cmd = cat_cmd(cmd_train, cmd_cpresult, cmd_accuracy, cmd_csv)
os.system(cmd)


import os
import shutil
import glob
import sys

path='../../aegnn_results/training_results/latest'
if not os.path.exists(path):
    os.makedirs(path)
else:
    # clean
    try:
        shutil.rmtree(path)
    except OSError as e:
        print("Error: %s : %s" % (path, e.strerror))
    
    # rebuild
    os.makedirs(path)

src_model = sorted(glob.glob(r'/space/yyang22/datasets/data/scratch/checkpoints/ncars/recognition/*/*.pt'), key=os.path.getctime)[-1]
dst_model = os.path.join(path,'latest_model.pt')

src_log = sorted(glob.glob(r'/space/yyang22/datasets/data/scratch/debug/*'), key=os.path.getctime)[-1]
dst_log = os.path.join(path,os.path.basename(src_log))


try:
    shutil.copy2(src_model, dst_model)
except IOError as e:
    print("Unable to copy file. %s" % e)
except:
    print("Unexpected error:", sys.exc_info())

try:
    shutil.copy2(src_log, dst_log)
except IOError as e:
    print("Unable to copy file. %s" % e)
except:
    print("Unexpected error:", sys.exc_info())
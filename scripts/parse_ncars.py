import struct
import os
import shutil
from tqdm import tqdm
import random


def load_atis_data(filename, flipX=0, flipY=0):
    td_data = {'ts': [], 'x': [], 'y': [], 'p': []}
    header = []

    with open(filename, 'rb') as f:
        # Parse header if any
        endOfHeader = False
        numCommentLine = 0
        while not endOfHeader:
            bod = f.tell()
            tline = f.readline().decode('utf-8', errors='ignore')
            if tline[0] != '%':
                endOfHeader = True
            else:
                words = tline.split()
                if len(words) > 2:
                    if words[1] == 'Date':
                        if len(words) > 3:
                            header.append((words[1], words[2] + ' ' + words[3]))
                    else:
                        header.append((words[1], words[2]))
                numCommentLine += 1
        f.seek(bod)

        evType = 0
        evSize = 8
        if numCommentLine > 0:  # Ensure compatibility with previous files.
            # Read event type
            evType = struct.unpack('b', f.read(1))[0]
            # Read event size
            evSize = struct.unpack('b', f.read(1))[0]

        bof = f.tell()
        f.seek(0, 2)
        numEvents = (f.tell() - bof) // evSize

        # Read data
        f.seek(bof)  # Start just after the header
        for _ in range(numEvents):
            timestamp = struct.unpack('<I', f.read(4))[0]
            timestamp *= 1e-6  # us -> s
            addr = struct.unpack('<I', f.read(4))[0]
            x = (addr & 0x00003FFF) >> 0
            y = (addr & 0x0FFFC000) >> 14
            p = (addr & 0x10000000) >> 28

            td_data['ts'].append(timestamp)
            td_data['x'].append(x if flipX == 0 else flipX - x)
            td_data['y'].append(y if flipY == 0 else flipY - y)
            td_data['p'].append(p)

    return td_data, header


# output
source_folder_train = '/space/yyang22/datasets/data/storage/original_ncars/Prophesee_Dataset_n_cars/n-cars_train'
source_folder_test = '/space/yyang22/datasets/data/storage/original_ncars/Prophesee_Dataset_n_cars/n-cars_test'
ttshuffled = '/space/yyang22/datasets/data/storage/ncars/ttshuffled'


sequence_counter = 0
train_counter = 0
# traverse binary files in NCars
for root, dirs, files in os.walk(source_folder_train):
    for dir_name in dirs:
        subfolder = os.path.join(root, dir_name)

        if 'background' in subfolder:
            is_car = 0
        elif 'cars' in subfolder:
            is_car = 1
        else:
            continue


        for file_name in tqdm(os.listdir(subfolder), desc=dir_name):
            if file_name.endswith('.dat'):
                binary_file = os.path.join(subfolder, file_name)

                td_data, _ = load_atis_data(binary_file)

                # make events.txt
                sequence_folder = os.path.join(ttshuffled, f'sequence_{sequence_counter}')
                os.makedirs(sequence_folder, exist_ok=True)

                events_file = os.path.join(sequence_folder, 'events.txt')
                with open(events_file, 'w') as txt_file:
                    for i in range(len(td_data['ts'])):
                        formatted_line = "{:.18e} {:.18e} {:.18e} {:.18e}".format(td_data['x'][i], td_data['y'][i], td_data['ts'][i], td_data['p'][i])
                        txt_file.write(formatted_line + '\n')

                # make is_car.txt
                is_car_file = os.path.join(sequence_folder, 'is_car.txt')
                with open(is_car_file, 'w') as txt_file:
                    txt_file.write(str(is_car))

                # add counter
                sequence_counter += 1
                train_counter += 1
print(f'Parsed {train_counter} ncars samples for training')

test_counter = 0
# traverse binary files in NCars
for root, dirs, files in os.walk(source_folder_test):
    for dir_name in dirs:
        subfolder = os.path.join(root, dir_name)

        if 'background' in subfolder:
            is_car = 0
        elif 'cars' in subfolder:
            is_car = 1
        else:
            continue


        for file_name in tqdm(os.listdir(subfolder), desc=dir_name):
            if file_name.endswith('.dat'):
                binary_file = os.path.join(subfolder, file_name)

                td_data, _ = load_atis_data(binary_file)

                # make events.txt
                sequence_folder = os.path.join(ttshuffled, f'sequence_{sequence_counter}')
                os.makedirs(sequence_folder, exist_ok=True)

                events_file = os.path.join(sequence_folder, 'events.txt')
                with open(events_file, 'w') as txt_file:
                    for i in range(len(td_data['ts'])):
                        formatted_line = "{:.18e} {:.18e} {:.18e} {:.18e}".format(td_data['x'][i], td_data['y'][i], td_data['ts'][i], td_data['p'][i])
                        txt_file.write(formatted_line + '\n')

                # make is_car.txt
                is_car_file = os.path.join(sequence_folder, 'is_car.txt')
                with open(is_car_file, 'w') as txt_file:
                    txt_file.write(str(is_car))

                # add counter
                sequence_counter += 1
                test_counter += 1
print(f'Parsed {test_counter} ncars samples for test')
print(f'Total {train_counter+test_counter} samples parsed; {len(os.listdir(ttshuffled))} files in {ttshuffled}')



# Random re-assign
training_count = 14417
validation_count = 4806
test_count = 4806
print(f'Now random re-assigning {training_count} for training, {validation_count} for val, and {test_count} for test')
seed = 12345
random.seed(seed)
print(f'Seed {seed}')


source_folder = ttshuffled
output_folder = '/space/yyang22/datasets/data/storage/ncars'

all_files = os.listdir(source_folder)
random.shuffle(all_files)


subfolders = ["training", "validation", "test"]
for subfolder in subfolders:
    subfolder_path = os.path.join(output_folder, subfolder)
    os.makedirs(subfolder_path, exist_ok=False)


for i, subfolder in enumerate(subfolders):
    count = training_count if i == 0 else validation_count if i == 1 else test_count
    for j in tqdm(range(count), desc=subfolder):
        if not all_files:
            break
        source_file = all_files.pop()
        source_file_path = os.path.join(source_folder, source_file)
        dest_file_path = os.path.join(output_folder, subfolder, source_file)
        shutil.move(source_file_path, dest_file_path)


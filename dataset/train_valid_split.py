from pathlib import Path
import random
import shutil

train_size = 0.8

#@ roboflow dir
rf_data_dir = Path('rf_data')
rf_label_dir = Path("rf_label")

#@ tạo folder image
train_dir = Path("images/train")
valid_dir = Path("images/valid")
train_label_dir = Path("labels/train")
valid_label_dir = Path('labels/valid')

if train_dir.exists():
    shutil.rmtree(train_dir)
if valid_dir.exists():
    shutil.rmtree(valid_dir)
train_dir.mkdir(parents= True, exist_ok= True)
valid_dir.mkdir(parents=True, exist_ok= True)

if train_label_dir.exists():
    shutil.rmtree(train_label_dir)
if valid_label_dir.exists():
    shutil.rmtree(valid_label_dir)
train_label_dir.mkdir(parents= True, exist_ok= True)
valid_label_dir.mkdir(parents=True, exist_ok= True)

images_dirs = [
    img_dir for img_dir in rf_data_dir.iterdir()
    if img_dir.is_file() and img_dir.suffix in ['.jpg', '.jpeg']
]
labels_dirs = [
    label_dir for label_dir in rf_label_dir.iterdir()
    if label_dir.is_file() and label_dir.suffix == '.txt'
]

#@ shuffle data
# random.shuffle(images_dirs)

num_train = int(len(images_dirs) * train_size)

for i in range(num_train):
    shutil.copy(labels_dirs[i], train_label_dir / Path(labels_dirs[i].name))
    shutil.copy(images_dirs[i], train_dir / Path(images_dirs[i].name))
for i in range(num_train, len(images_dirs)):
    shutil.copy(labels_dirs[i], valid_label_dir / Path(labels_dirs[i].name))
    shutil.copy(images_dirs[i], valid_dir / Path(images_dirs[i].name))
    
# %%
import os
from datetime import datetime

import torch.utils.data
from torch import optim
# from sklearn.model_selection import train_test_split
from tqdm import tqdm

from datasets import load_dataset
from torch.utils.data import DataLoader

from torchvision import transforms
from minimagen.Imagen import Imagen
from minimagen.Unet import Unet, Base, Super, BaseTest, SuperTest
from minimagen.generate import load_minimagen, load_params
from minimagen.t5 import get_encoded_dim
from minimagen.training import get_minimagen_parser, ConceptualCaptions, get_minimagen_dl_opts, \
    create_directory, get_model_params, get_model_size, save_training_info, get_default_args, MinimagenTrain, \
    load_restart_training_parameters, load_testing_parameters
def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

# %%

dataset_name = "lambdalabs/pokemon-blip-captions"
IMG_SIDE_LEN = 64
batch = 4

to_tensor = transforms.Compose([
    transforms.Resize(IMG_SIDE_LEN),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
])
def preprocess(data):
    for i in range(len(data['image'])):
        data['image'][i] = to_tensor(data['image'][i])
    return data
dataset = load_dataset(dataset_name, split="train", cache_dir='.').train_test_split(test_size=0.1)
print(f"dataset: {dataset}")
train_dataset = dataset['train']
test_dataset = dataset['test']
train_dataset = train_dataset.with_transform(preprocess)
test_dataset = test_dataset.with_transform(preprocess)

print(f"train_dataset: {train_dataset}")
train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, drop_last=True)
loader = sample_data(train_dataloader)

# sample_text_img = next(loader)
# text = sample_text_img['text'][:1]
# print(f"text: {text}")

for batch_num, batch in tqdm(enumerate(train_dataloader)):
    # print(batch['text'][:1])
    print(batch[0])

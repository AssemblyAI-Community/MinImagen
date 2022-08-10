import json
import os
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from functools import partial
import io
import urllib
from typing import Literal
from tqdm import tqdm

import datasets
import PIL.Image
from einops import rearrange
import torch.utils.data
from torch import optim
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor

from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent
from resize_right import resize

from minimagen.Imagen import Imagen
from minimagen.Unet import Unet
from minimagen.helpers import exists
from minimagen.t5 import get_encoded_dim, t5_encode_text
from minimagen.generate import load_minimagen

USER_AGENT = get_datasets_user_agent()


class MinimagenDataset(torch.utils.data.Dataset):

    def __init__(self, hf_dataset, *, encoder_name, max_length, train=True, img_transform=None):
        """
        MinImagen Dataset

        :param hf_dataset: HuggingFace `datasets` Dataset object
        :param encoder_name: Name of the T5 encoder to use.
        :param max_length: Maximum number of words allowed in a given caption.
        :param train: Whether train or test dataset
        :param img_transform: (optional) Transforms to be applied on a sample
        """

        split = "train" if train else "validation"

        self.urls = hf_dataset[f"{split}"]['image_url']
        self.captions = hf_dataset[f"{split}"]['caption']

        self.img_transform = img_transform
        self.encoder_name = encoder_name
        self.max_length = max_length

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = fetch_single_image(self.urls[idx])
        if img is None:
            return None
        elif self.img_transform:
            img = self.img_transform(img)

        # Have to check None again because `Resize` transform can return None
        if img is None:
            return None
        elif img.shape[0] != 3:
            return None

        enc, msk = t5_encode_text([self.captions[idx]], self.encoder_name, self.max_length)

        return {'image': img, 'encoding': enc, 'mask': msk}


class Rescale:
    """
    Transformation to scale images to the proper size
    """

    def __init__(self, side_length):
        self.side_length = side_length

    def __call__(self, sample, *args, **kwargs):
        if len(sample.shape) == 2:
            sample = rearrange(sample, 'h w -> 1 h w')
        elif not len(sample.shape) == 3:
            raise ValueError("Improperly shaped image for rescaling")

        sample = resize_image_to_square(sample, self.side_length)

        # If there was an error in the resizing, return None
        if sample is None:
            return None

        # Rescaling max push images out of [0,1] range, so have to standardize:
        sample -= sample.min()
        sample /= sample.max()
        return sample


def collate(batch):
    # Filter out None instances or those in which the image could not be fetched
    batch = list(filter(lambda x: x is not None, batch))
    batch = list(filter(lambda x: x['image'] is not None, batch))

    # If the batch is empty after filtering
    if not batch:
        return None

    # Expand mask and encodings to len of elt in batch with greatest number of words
    max_len = max([batch[i]['mask'].shape[1] for i in range(len(batch))])

    for elt in batch:
        length = elt['mask'].shape[1]
        rem = max_len - length
        elt['mask'] = torch.squeeze(elt['mask'])
        elt['encoding'] = torch.squeeze(elt['encoding'])
        if rem > 0:
            elt['mask'] = F.pad(elt['mask'], (0, rem), 'constant', 0)
            elt['encoding'] = F.pad(elt['encoding'], (0, 0, 0, rem), 'constant', False)

    # TODO: Should really be passing in `device` - find a more elegant way to do this
    for didx, datum in enumerate(batch):
        for tensor in datum.keys():
            batch[didx][tensor] = batch[didx][tensor].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    return torch.utils.data.dataloader.default_collate(batch)


def fetch_images(batch, num_threads, timeout=None, retries=0):
    fetch_single_image_with_args = partial(fetch_single_image, timeout=timeout, retries=retries)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch["image"] = list(executor.map(fetch_single_image_with_args, batch["image_url"]))
    return batch


def fetch_single_image(image_url, timeout=None, retries=0):
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = PIL.Image.open(io.BytesIO(req.read()))
            break
        except Exception:
            image = None
    return image


def resize_image_to_square(image: torch.tensor,
                    target_image_size: int,
                    clamp_range: tuple = None,
                    pad_mode: Literal['constant', 'edge', 'reflect', 'symmetric'] = 'reflect'
                    ) -> torch.tensor:
    """
    Resizes image to desired size.

    :param image: Images to resize. Shape (b, c, s, s)
    :param target_image_size: Edge length to resize to.
    :param clamp_range: Range to clamp values to. Tuple of length 2.
    :param pad_mode: `constant`, `edge`, `reflect`, `symmetric`.
        See [TorchVision documentation](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.pad.html) for additional details
    :return: Resized image. Shape (b, c, target_image_size, target_image_size)
    """
    h_scale = image.shape[-2]
    w_scale = image.shape[-1]

    if h_scale == target_image_size and w_scale == target_image_size:
        return image

    scale_factors = (target_image_size / h_scale, target_image_size / w_scale)
    try:
        out = resize(image, scale_factors=scale_factors, pad_mode=pad_mode)
    except:
        return None

    if exists(clamp_range):
        out = out.clamp(*clamp_range)

    return out

def create_directory(dir_path):
    """
    Creates a directory at the given path if it does not exist already and returns a context manager that allows user
        to temporarily enter the direcory
    """
    original_dir = os.getcwd()
    if not os.path.exists(dir_path):
        # TODO: makedirs make intermediate dirs. Try for i in [param, dic, tmp] os.make...
        os.makedirs(dir_path)
        os.makedirs(os.path.join(dir_path, "parameters"))
        os.makedirs(os.path.join(dir_path, "state_dicts"))
        os.makedirs(os.path.join(dir_path, "tmp"))

    @contextmanager
    def cm(subpath=""):
        os.chdir(os.path.join(dir_path, subpath))
        yield
        os.chdir(original_dir)
    return cm


def make_params(parameters_dir):
    im_params = None
    unets_params = []

    # Find appropriate files
    for file in os.listdir(parameters_dir):
        if file.startswith('imagen'):
            im_params = file
        elif file.startswith('unet_'):
            unets_params.append(file)

    # Make sure UNets params are sorted properly
    unets_params = sorted(unets_params, key=lambda x: int(x.split('_')[1]))

    return unets_params, im_params


# Get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create training directory
dir_path = f"./training_{timestamp}"
training_dir = create_directory(dir_path)

# Optionally hard-code values (will overwrite command-line)
PARAMETERS = None
NUM_WORKERS = None
BATCH_SIZE = None
MAX_NUM_WORDS = None
IMG_SIDE_LEN = None
EPOCHS = None
T5_NAME = None
TRAIN_VALID_FRAC = None
TIMESTEPS = None
OPTIM_LR = None
ACCUM_ITER = None
CHCKPT_FRAC = None
TESTING = None

# Command line argument parser
parser = ArgumentParser()
parser.add_argument("-p", "--PARAMETERS", dest="PARAMETERS", help="Parameters directory", default=None, type=str)
parser.add_argument("-n", "--NUM_WORKERS", dest="NUM_WORKERS", help="Number of workers for DataLoader", default=0, type=int)
parser.add_argument("-b", "--BATCH_SIZE", dest="BATCH_SIZE", help="Batch size", default=2, type=int)
parser.add_argument("-mw", "--MAX_NUM_WORDS", dest="MAX_NUM_WORDS", help="Maximum number of words allowed in a caption", default=64, type=int)
parser.add_argument("-s", "--IMG_SIDE_LEN", dest="IMG_SIDE_LEN", help="Side length of square Imagen output images", default=128, type=int)
parser.add_argument("-e", "--EPOCHS", dest="EPOCHS", help="Number of training epochs", default=5, type=int)
parser.add_argument("-t5", "--T5_NAME", dest="T5_NAME", help="Name of T5 encoder to use", default='t5_base', type=str)
parser.add_argument("-f", "--TRAIN_VALID_FRAC", dest="TRAIN_VALID_FRAC", help="Fraction of dataset to use for training (vs. validation)", default=0.8, type=float)
parser.add_argument("-t", "--TIMESTEPS", dest="TIMESTEPS", help="Number of timesteps in Diffusion process", default=1000, type=int)
parser.add_argument("-lr", "--OPTIM_LR", dest="OPTIM_LR", help="Learning rate for Adam optimizer", default=0.0001, type=float)
parser.add_argument("-ai", "--ACCUM_ITER", dest="ACCUM_ITER", help="Number of batches for gradient accumulation", default=1, type=int)
parser.add_argument("-cf", "--CHCKPT_FRAC", dest="CHCKPT_FRAC", help="Percentage of way through training epoch to checkpoint at", default=0.1, type=float)
parser.add_argument("-test", "--TESTING", dest="TESTING", help="Whether to test with smaller dataset", default=False, type=bool)
args = parser.parse_args()

# For each command-line argument, replace with hard-coded value if it exists
for i in args.__dict__.keys():
    if vars()[i] is not None and vars()[i] != getattr(args, i):
        print(f"\nWARNING: {i} defaulting to hard-coded value of {vars()[i]} rather than the value of {getattr(args, i)} "
              f"supplied by a (potentially default) command-line argument. Edit `./train.py` to remove this value.\n")
    else:
        vars()[i] = getattr(args, i)

# HuggingFace dataset
dset = load_dataset("conceptual_captions")

# If testing, lower parameter values for lower computational load and also lower amount of data being used.
if TESTING:
    BATCH_SIZE = 2
    MAX_NUM_WORDS = 32
    IMG_SIDE_LEN = 128
    EPOCHS = 2
    T5_NAME = 't5_base'
    TRAIN_VALID_FRAC = 0.5
    TIMESTEPS = 25  # Do not make less than 20
    OPTIM_LR = 0.0001

    num = 16
    vi = dset['validation']['image_url'][:num]
    vc = dset['validation']['caption'][:num]
    ti = dset['train']['image_url'][:num]
    tc = dset['train']['caption'][:num]
    dset = datasets.Dataset = {'train':{
                'image_url': ti,
                'caption': tc,
                    }, 'num_rows':num,
        'validation':{
            'image_url': ti,
            'caption': tc,}, 'num_rows':num}

# Save the training parameters
with training_dir("parameters"):
    with open(f"training_parameters_{timestamp}.txt", "w") as f:
        for i in args.__dict__.keys():
            f.write(f'--{i}={vars()[i]}\n')

# Torch train/valid dataset
dataset_train_valid = MinimagenDataset(dset, max_length=MAX_NUM_WORDS, encoder_name=T5_NAME, train=True,
                                       img_transform=Compose([ToTensor(), Rescale(IMG_SIDE_LEN)]))

# Split into train/valid
train_size = int(TRAIN_VALID_FRAC * len(dataset_train_valid))
valid_size = len(dataset_train_valid) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(dataset_train_valid, [train_size, valid_size])

# Torch test dataset
test_dataset = MinimagenDataset(dset, max_length=MAX_NUM_WORDS, train=False, encoder_name=T5_NAME,
                                img_transform=Compose([ToTensor(), Rescale(IMG_SIDE_LEN)]))

# Create dataloaders
dl_opts = {'batch_size': BATCH_SIZE,
           'shuffle': False,
           'num_workers': NUM_WORKERS,
           'drop_last':True,
           'collate_fn':collate}

train_dataloader = torch.utils.data.DataLoader(train_dataset, **dl_opts)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, **dl_opts)
test_dataloader = torch.utils.data.DataLoader(test_dataset, **dl_opts)

# Get encoding dimension of the text encoder
text_embed_dim = get_encoded_dim(T5_NAME)

imagen = load_minimagen(os.path.join(os.getcwd(), "training_20220808_190825"))

param_size = 0
for param in imagen.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in imagen.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2

with training_dir():
    with open('training_progess.txt', 'a') as f:
        f.write(f'STARTED FROM CHECKPOINT 190825\nmodel size: {size_all_mb:.3f}MB\n\n\n')

optimizer = optim.Adam(imagen.parameters(), lr=OPTIM_LR)
print("Created optimzer")

# Train
best_loss = [9999999 for i in range(2)]


import signal, time

class Timeout():
  """Timeout class using ALARM signal"""
  class Timeout(Exception): pass

  def __init__(self, sec):
    self.sec = sec

  def __enter__(self):
    signal.signal(signal.SIGALRM, self.raise_timeout)
    signal.alarm(self.sec)

  def __exit__(self, *args):
    signal.alarm(0) # disable alarm

  def raise_timeout(self, *args):
    raise Timeout.Timeout()

# Run block of code with timeouts





for epoch in range(EPOCHS):
    print(f'\n{"-"*20} EPOCH {epoch+1} {"-"*20}')
    with training_dir():
        with open('training_progess.txt', 'a') as f:
            f.write(f'{"-"*20} EPOCH {epoch+1} {"-"*20}\n')

    imagen.train(True)

    running_train_loss = [0. for i in range(2)]
    print('Training...')

    for batch_num, batch in tqdm(enumerate(train_dataloader)):
        try:
            with Timeout(60*5):

                # If batch is empty, move on to the next one
                if not batch:
                    continue

                images = batch['image']
                encoding = batch['encoding']
                mask = batch['mask']

                losses = [0. for i in range(2)]
                for unet_idx in range(2):
                    loss = imagen(images, text_embeds=encoding, text_masks=mask, unet_number=unet_idx+1)
                    running_train_loss[unet_idx] += loss.detach()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(imagen.parameters(), 50)

                # Gradient accumulation optimizer step (first bool for logical short-circuiting)
                if ACCUM_ITER == 1 or (batch_num % ACCUM_ITER == 0) or (batch_num + 1 == len(train_dataloader)):
                    optimizer.step()
                    optimizer.zero_grad()

                # Every 10% of the way through epoch, save states in case of training failure
                if batch_num % (len(train_dataloader)*CHCKPT_FRAC) == 0 or (batch_num % 1000 == 0 and batch_num <= 10000):
                    with training_dir("tmp"):
                        for idx in range(2):
                            model_path = f"unet_{idx}_{int(batch_num/len(train_dataloader))*100}_percent.pth"
                            torch.save(imagen.unets[idx].state_dict(), model_path)

                    with training_dir():
                        with open('training_progess.txt', 'a') as f:
                            f.write(f'Checkpoint created at batch number {batch_num}\n')
        except Timeout.Timeout:
            pass

    avg_loss = [i / len(train_dataloader) for i in running_train_loss]
    with training_dir():
        with open('training_progess.txt', 'a') as f:
            f.write(f'U-Nets Avg Train Losses: {[round(i.item(), 3) for i in avg_loss]}\n')

    # Compute average loss across validation batches for each unet
    running_valid_loss = [0. for i in range(2)]
    imagen.train(False)
    print('Validating...')
    for vbatch in tqdm(valid_dataloader):
        # If batch is empty, move on to the next one
        if not batch:
            continue

        images = vbatch['image']
        encoding = vbatch['encoding']
        mask = vbatch['mask']

        for unet_idx in range(2):
            running_valid_loss[unet_idx] += imagen(images, text_embeds=encoding, text_masks=mask, unet_number=unet_idx + 1).detach()

    avg_loss = [i/len(valid_dataloader) for i in running_valid_loss]
    with training_dir():
        with open('training_progess.txt', 'a') as f:
            f.write(f'U-Nets Avg Valid Losses: {[round(i.item(), 3) for i in avg_loss]}\n\n')

    # If validation loss less than previous best, save the model weights
    for i, l in enumerate(avg_loss):
        print(f"Unet {i} avg validation loss: ", l)
        if l < best_loss[i]:
            best_loss[i] = l
            with training_dir("state_dicts"):
                model_path = f"unet_{i}_state_{timestamp}_epoch_{epoch+1}_{l:.3f}.pth"
                torch.save(imagen.unets[i].state_dict(), model_path)

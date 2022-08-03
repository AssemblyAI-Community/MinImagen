from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
import io
import urllib
from typing import Literal

import datasets
import nonechucks as nc
import PIL.Image
from einops import rearrange
import torch.utils.data
from torch import optim
from torchvision.transforms import Compose, ToTensor

from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent
from resize_right import resize

from minimagen.Imagen import Imagen
from minimagen.Unet import Unet
from minimagen.helpers import exists
from minimagen.t5 import get_encoded_dim, t5_encode_text

USER_AGENT = get_datasets_user_agent()


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


def fetch_images(batch, num_threads, timeout=None, retries=0):
    fetch_single_image_with_args = partial(fetch_single_image, timeout=timeout, retries=retries)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch["image"] = list(executor.map(fetch_single_image_with_args, batch["image_url"]))
    return batch

class Rescale:

    def __init__(self, side_length):
        self.side_length = side_length

    def __call__(self, sample, *args, **kwargs):
        '''
        if len(sample.shape) == 2:
            sample = rearrange(sample, 'h w -> 1 1 h w')
        elif len(sample.shape) == 3:
            sample = rearrange(sample, 'c h w -> 1 c h w')
        elif not len(sample.shape) == 4:
            raise ValueError("Improperly shaped image for rescaling")
        '''
        if len(sample.shape) == 2:
            sample = rearrange(sample, 'h w -> 1 h w')
        elif not len(sample.shape) == 3:
            raise ValueError("Improperly shaped image for rescaling")

        return resize_image_to_square(sample, self.side_length)

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
    out = resize(image, scale_factors=scale_factors, pad_mode=pad_mode)

    if exists(clamp_range):
        out = out.clamp(*clamp_range)

    return out


class MinimagenDataset(torch.utils.data.Dataset):

    def __init__(self, hf_dataset, *, encoder_name, max_length, train = True, img_transform = None):
        """
        MinImagen Dataset

        :param hf_dataset: HuggingFace `datasets` Dataset object
        :param train: Whether train or test dataset
        :param transform: (optional) Transforms to be applied on a sample
        """

        split = "train" if train else "validation"

        self.urls = dset[f"{split}"]['image_url']
        self.encoding, self.mask = t5_encode_text(dset[f"{split}"]['caption'], encoder_name, max_length)
        self.img_transform = img_transform

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
        if img.shape[0] != 3:
            return None

        return {'image': img, 'encoding': self.encoding[idx], 'mask': self.mask[idx]}

def collate(batch):
    batch = list(filter(lambda x: x['image'] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

# Constants
BATCH_SIZE = 4  # Batch size training data
MAX_NUM_WORDS = 64  # Max number of words allowed in a caption
IMG_SIDE_LEN = 128  # Side length of the training images/final output image from Imagen
EPOCHS = 5  # Number of epochs to train from
T5_NAME = "t5_small"  # Name of the T5 encoder to use
TRAIN_VALID_FRAC = 0.5 # TODO: Change to 0.8
TIMESTEPS = 250
OPTIM_LR = 0.0001
TESTING = True

# Get encoding dimension of the text encoder
text_embed_dim = get_encoded_dim(T5_NAME)

# HuggingFace dataset
dset = load_dataset("conceptual_captions")
# Cut down size for testing
if TESTING:
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
    print(dset['train']['caption'])

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
# Safe dataloaders
dl_opts = {'batch_size': BATCH_SIZE, 'shuffle': False, 'num_workers': 0, 'drop_last':True, 'collate_fn':collate}

# Problems with SafeDataLoaders - may need to downgrade to torch 1.2.0
#train_dataloader = nc.SafeDataLoader(nc.SafeDataset(train_dataset), **dl_opts)
#valid_dataloader = nc.SafeDataLoader(nc.SafeDataset(valid_dataset), **dl_opts)
#test_dataloader = nc.SafeDataLoader(nc.SafeDataset(test_dataset), **dl_opts)


train_dataloader = torch.utils.data.DataLoader(nc.SafeDataset(train_dataset), **dl_opts)
valid_dataloader = torch.utils.data.DataLoader(nc.SafeDataset(valid_dataset), **dl_opts)
test_dataloader = torch.utils.data.DataLoader(nc.SafeDataset(test_dataset), **dl_opts)

# Value min is -0.0461 and max is 1.0819 - should either be in [0,1] or [-1,1]
#for batch in train_dataloader:
#    print(torch.min(batch['image']), torch.max(batch['image']))

# Get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create Unets
base_unet = Unet(
    dim=128,
    text_embed_dim=text_embed_dim,
    cond_dim=64,
    dim_mults=(1, 2, 4),
    num_resnet_blocks=2,
    layer_attns=(False, False, True),
    layer_cross_attns=(False, False, True),
    attend_at_middle=True
)

super_res_unet = Unet(
    dim=256,
    text_embed_dim=text_embed_dim,
    cond_dim=512,
    dim_mults=(1, 2, 4),
    num_resnet_blocks=(2, 4, 8),
    layer_attns=(False, False, True),
    layer_cross_attns=(False, False, True),
    attend_at_middle=False
)
unets = (base_unet, super_res_unet)
print("Created Unets")

# Create Imagen from Unets
imagen = Imagen(
    unets=unets,
    image_sizes=(32, 128),
    timesteps=TIMESTEPS,  # has to be at least 20.
    cond_drop_prob=0.1
).to(device)
print("Created Imagen")

optimizer = optim.Adam(imagen.parameters(), lr=OPTIM_LR)
print("Created optimzer")

best_loss = [9999999 for i in range(len(unets))]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
for epoch in range(EPOCHS):
    print(f'\nEPOCH {epoch+1}\n')

    imagen.train(True)

    avg_loss = None
    for batch in train_dataloader:
        images = batch['image']
        encoding = batch['encoding']
        mask = batch['mask']

        losses = [0. for i in range(len(unets))]
        for unet_idx in range(len(unets)):
            optimizer.zero_grad()
            loss = imagen(images, text_embeds=encoding, text_masks=mask, unet_number=unet_idx+1)
            losses[unet_idx] += loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(imagen.parameters(), 50)
            optimizer.step()
        print(f'LOSSES: {losses}')

    # Compute average loss across validation batches for each unet
    running_vloss = [0. for i in range(len(unets))]
    imagen.train(False)
    for vbatch in valid_dataloader:
        for unet_idx in range(len(unets)):
            vimages = vbatch['image']
            vencoding = vbatch['encoding']
            vmask = vbatch['mask']

            running_vloss[unet_idx] += imagen(vimages, text_embeds=vencoding, text_masks=vmask, unet_number=unet_idx + 1)
    avg_loss = [i/len(valid_dataloader) for i in running_vloss]

    # If validation loss less than previous best, save the model weights
    for i, l in enumerate(avg_loss):
        print(f"Unet {i} avg validation loss: ", l)
        if l < best_loss[i]:
            best_loss[i] = l
        model_path = f"imagen_{timestamp}_{i}_{epoch+1}.pth"
        torch.save(imagen.unets[i].state_dict(), model_path)





# Generate images with "trained" model
#print("Sampling from Imagen...")
#images = imagen.sample(texts=CAPTIONS, cond_scale=3., return_pil_images=True)

# Save output PIL images
#print("Saving Images")
#for idx, img in enumerate(images):
#    img.save(f'Generated_Image_{idx}.png')

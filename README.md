# MinImagen
### A Minimal implementation of the [Imagen](https://imagen.research.google/) text-to-image model.

<p align="center"><img src="./images/model_structure.png?raw=True" width="700"/></p>

### For a tutorial on building this model, see [here](www.assemblyai.com/blog/build-your-own-imagen-text-to-image-model/).

Given a caption of an image, Imagen will generate an image that reflects the caption. The model is a simple [cascading diffusion model](https://arxiv.org/abs/2106.15282), using a T5 text encoder to encode the captions which conditions a base image generator, and then a sequence of super-resolution models.

In particular, two notable contributions are the developments of:
1. [**Noise Conditioning Augmentation**](https://www.assemblyai.com/blog/how-imagen-actually-works/#robust-cascaded-diffusion-models), which noises low-resolution conditioning images in the super-resolution models, and
2. [**Dynamic Thresholding**](https://www.assemblyai.com/blog/how-imagen-actually-works/#dynamic-thresholding) which helps prevent image saturation at high [classifier-free guidance](https://www.assemblyai.com/blog/how-imagen-actually-works/#classifier-free-guidance) weights.

<br/>

**See [How Imagen Actually Works](https://www.assemblyai.com/blog/how-imagen-actually-works/) for a detailed explanation of Imagen's operating principles.**

<br/>


## Attribution Note
This implementation is largely based on Phil Wang's [Imagen implementation](https://github.com/lucidrains/imagen-pytorch).

## Installation
To install MinImagen, run the following command in the terminal:
```bash
$ pip install minimagen
```
**Note that MinImagen requires Python3.9 or higher**
## Documentation
Documentation can be found [here](https://assemblyai-examples.github.io/MinImagen/)

## Usage

A minimal usage:
```python
import os
from datetime import datetime

import torch.utils.data
from torch import optim

from minimagen.Imagen import Imagen
from minimagen.Unet import Unet, Base, Super, BaseTest, SuperTest
from minimagen.generate import load_minimagen, load_params
from minimagen.t5 import get_encoded_dim
from minimagen.training import get_minimagen_parser, ConceptualCaptions, get_minimagen_dl_opts, \
    create_directory, get_model_params, get_model_size, save_training_info, get_default_args, MinimagenTrain, \
    load_restart_training_parameters, load_testing_parameters

# Get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Command line argument parser
parser = get_minimagen_parser()
args = parser.parse_args()

# Create training directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
dir_path = f"./training_{timestamp}"
training_dir = create_directory(dir_path)

# Load [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/) dataset. 
# If testing a training script, replace some command line args with less computationally demanding values.
if args.TESTING:
    args = load_testing_parameters(args)
    train_dataset, valid_dataset = ConceptualCaptions(args, smalldata=True)
else:
    train_dataset, valid_dataset = ConceptualCaptions(args, smalldata=False)

# Create dataloaders
dl_opts = {**get_minimagen_dl_opts(device), 'batch_size': args.BATCH_SIZE, 'num_workers': args.NUM_WORKERS}
train_dataloader = torch.utils.data.DataLoader(train_dataset, **dl_opts)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, **dl_opts)

# Get encoding dimension of the text encoder
text_embed_dim = get_encoded_dim(args.T5_NAME)

# If testing a training script, use small U-Nets to lower computational load.
if args.TESTING:
    # If testing, use tiny MinImagen for low computational load
    unets_params = [get_default_args(BaseTest), get_default_args(SuperTest)]

# Otherwise, load U-Net parameters from original Imagen implementation
else:
    # If no parameters provided, use params from minimagen.Imagen.Base and minimagen.Imagen.Super built-in classes
    unets_params = [get_default_args(Base), get_default_args(Super)]

# Create Unets accoridng to unets_params
unets = [Unet(**unet_params).to(device) for unet_params in unets_params]

# Specify MinImagen parameters
imagen_params = dict(
    image_sizes=(int(args.IMG_SIDE_LEN / 2), args.IMG_SIDE_LEN),
    timesteps=args.TIMESTEPS,
    cond_drop_prob=0.15,
    text_encoder_name=args.T5_NAME
)

# Create MinImagen from UNets with specified imagen parameters
imagen = Imagen(unets=unets, **imagen_params).to(device)

# Fill in unspecified arguments with defaults to record complete config (parameters) file
unets_params = [{**get_default_args(Unet), **i} for i in unets_params]
imagen_params = {**get_default_args(Imagen), **imagen_params}

# Get the size of the Imagen model in megabytes
model_size_MB = get_model_size(imagen)

# Save all training info (config files, model size, etc.)
save_training_info(args, timestamp, unets_params, imagen_params, model_size_MB, training_dir)

# Create optimizer
optimizer = optim.Adam(imagen.parameters(), lr=args.OPTIM_LR)

# Train the MinImagen instance
MinimagenTrain(timestamp, args, unets, imagen, train_dataloader, valid_dataloader, training_dir, optimizer, timeout=30)
```


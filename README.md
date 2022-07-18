# MinImagen
### A Minimal implementation of the [Imagen](https://imagen.research.google/) text-to-image model.

<p align="center"><img src="./images/model_structure.png?raw=True" width="700"/></p>

### For a tutorial on building this model, see [here](www.assemblyai.com/blog/build-your-own-imagen-text-to-image-model/).

Given a caption of an image, Imagen will generate an image that reflects the caption. The model is a simple [cascading diffusion model](https://arxiv.org/abs/2106.15282), using a T5 text encoder to encode the captions which conditions a base image generator, and then a sequence of super-resolution models.

In particular, two notable contributions are the developments of:
1. [Noise Conditioning Augmentation](https://www.assemblyai.com/blog/how-imagen-actually-works/#robust-cascaded-diffusion-models), which noises low-resolution conditioning images in the super-resolution models, and
2. [**Dynamic Thresholding**](https://www.assemblyai.com/blog/how-imagen-actually-works/#dynamic-thresholding) which helps prevent image saturation at high [classifier-free guidance](https://www.assemblyai.com/blog/how-imagen-actually-works/#classifier-free-guidance) weights.

<br/>

**See [How Imagen Actually Works](https://www.assemblyai.com/blog/how-imagen-actually-works/) for a detailed explanation of Imagen's operating principles.**

<br/>


## Attribution Note
This implementation is largely based on Phil Wang's [Imagen implementation](https://github.com/lucidrains/imagen-pytorch).

## Installation
```bash
$ pip install minimagen
```
## Documentation
Documentation can be found [here](https://assemblyai-examples.github.io/MinImagen/)

## Usage

A minimal usage:
```python
import torch
from minimagen.Imagen import Imagen
from minimagen.Unet import Unet, Base, Super
from minimagen.t5 import t5_encode_text, get_encoded_dim
from torch import optim

# Name of the T5 encoder to use
encoder_name = 't5_small'

# Text captions of training images
train_texts = [
    'a pepperoni pizza',
    'a man riding a horse',
    'a Beluga whale',
    'a woman rock climbing'
]

# Training images (side length equal to Imagen final output image size)
train_images = torch.randn(4, 3, 64, 64)

# Create the Imagen instance
enc_dim = get_encoded_dim(encoder_name)
unets = (Base(text_embed_dim=enc_dim), Super(text_embed_dim=enc_dim))
imagen = Imagen(unets=unets, image_sizes=(32, 64), timesteps=10)

# Create an optimzier
optimizer = optim.Adam(imagen.parameters())

# Train the U-Nets in Imagen
for j in range(10):
    for i in range(len(unets)):
        optimizer.zero_grad()
        loss = imagen(train_images, texts=train_texts, unet_number=i)
        loss.backward()
        optimizer.step()

# Sample captions to generate images for
sample_captions = [
    'a happy dog',
    'a big red house',
    'a woman standing on a beach',
    'a man on a bike'
]

# Generate images
images = imagen.sample(texts=sample_captions, cond_scale=3., return_pil_images=True)

# Save images
for idx, img in enumerate(images):
    img.save(f'Generated_Image_{idx}.png')
```

Text embeddings and masks can be precomputed, and Unets parameters can be specified rather than using `Base` and `Super`:

```python
train_encs, train_mask = t5_encode_text(train_texts, name=encoder_name)

enc_dim = get_encoded_dim(encoder_name)

base_unet = Unet(
    dim=32,
    text_embed_dim=enc_dim,
    cond_dim=64,
    dim_mults=(1, 2, 4),
    num_resnet_blocks=2,
    layer_attns=(False, False, True),
    layer_cross_attns=(False, False, True),
    attend_at_middle=True
)

super_res_unet = Unet(
    dim=32,
    text_embed_dim=enc_dim,
    cond_dim=512,
    dim_mults=(1, 2, 4),
    num_resnet_blocks=(2, 4, 8),
    layer_attns=(False, False, True),
    layer_cross_attns=(False, False, True),
    attend_at_middle=False
)

# Create Imagen instance
imagen = Imagen((base_unet, super_res_unet), image_sizes=(32, 64), timesteps=10)
```


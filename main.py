import torch
from torch import optim

from minimagen.Imagen import Imagen
from minimagen.Unet import Unet
from minimagen.t5 import get_encoded_dim

# Constants
BATCH_SIZE = 4  # Batch size training data
MAX_NUM_WORDS = 64  # Max number of words allowed in a caption
IMG_SIDE_LEN = 128  # Side length of the training images/final output image from Imagen
EPOCHS = 5  # Number of epochs to train from
T5_NAME = "t5_small"  # Name of the T5 encoder to use

# Captions to generate samples for
CAPTIONS = [
    'a happy dog',
    'a big red house',
    'a woman standing on a beach',
    'a man on a bike'
]

# Get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get encoding dimension of the text encoder
text_embed_dim = get_encoded_dim(T5_NAME)

# Create Unets
base_unet = Unet(
    dim=32,
    text_embed_dim=text_embed_dim,
    cond_dim=64,
    dim_mults=(1, 2, 4),
    num_resnet_blocks=2,
    layer_attns=(False, False, True),
    layer_cross_attns=(False, False, True),
    attend_at_middle=True
)

super_res_unet = Unet(
    dim=32,
    text_embed_dim=text_embed_dim,
    cond_dim=512,
    dim_mults=(1, 2, 4),
    num_resnet_blocks=(2, 4, 8),
    layer_attns=(False, False, True),
    layer_cross_attns=(False, False, True),
    attend_at_middle=False
)
print("Created Unets")

# Create Imagen from Unets
imagen = Imagen(
    unets=(base_unet, super_res_unet),
    image_sizes=(32, 128),
    timesteps=10,
    cond_drop_prob=0.1
).to(device)
print("Created Imagen")

# Create example data
text_embeds = torch.randn(
    BATCH_SIZE,
    MAX_NUM_WORDS,
    text_embed_dim).to(device)

text_masks = torch.ones(
    BATCH_SIZE,
    MAX_NUM_WORDS).bool().to(device)

images = torch.randn(
    BATCH_SIZE,
    3,
    IMG_SIDE_LEN,
    IMG_SIDE_LEN).to(device)
print("Created example data")

# Create optimizer
optimizer = optim.Adam(imagen.parameters())
print("Created optimzer")

# Train on example data
print("Training Imagen...")
for j in range(EPOCHS):
    for i in (1, 2):
        optimizer.zero_grad()
        loss = imagen(images, text_embeds=text_embeds, text_masks=text_masks, unet_number=i)
        loss.backward()
        optimizer.step()

# Generate images with "trained" model
print("Sampling from Imagen...")
images = imagen.sample(texts=CAPTIONS, cond_scale=3., return_pil_images=True)

# Save output PIL images
print("Saving Images")
for idx, img in enumerate(images):
    img.save(f'Generated_Image_{idx}.png')

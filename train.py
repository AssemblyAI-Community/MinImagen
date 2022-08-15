import os
import shutil
from datetime import datetime

import torch.utils.data
from torch import optim


from minimagen.Imagen import Imagen
from minimagen.Unet import Unet, Base, Super
from minimagen.generate import load_minimagen, load_params
from minimagen.t5 import get_encoded_dim
from minimagen.training import get_minimagen_parser, ConceptualCaptions, get_minimagen_dl_opts, \
    create_directory, get_model_params, get_model_size, save_training_info, get_default_args, MinimagenTrain, \
    load_restart_training_parameters

# Get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create training directory
dir_path = f"./training_{timestamp}"
training_dir = create_directory(dir_path)

# Command line argument parser
parser = get_minimagen_parser()
args = parser.parse_args()

if args.RESTART_DIRECTORY is not None:
    args.__dict__ = {**args.__dict__, **load_restart_training_parameters(args.RESTART_DIRECTORY)}

# If testing, lower parameter values for lower computational load and also lower amount of data being used.
if args.TESTING:
    # Parameters for testing - lowers computational load
    args.__dict__ = {
        **args.__dict__,
        **dict(
            BATCH_SIZE=2,
            MAX_NUM_WORDS=32,
            IMG_SIDE_LEN=128,
            EPOCHS=2,
            T5_NAME='t5_small',
            TRAIN_VALID_FRAC=0.5,
            TIMESTEPS=25,  # Do not make less than 20
            OPTIM_LR=0.0001
        )
    }
    train_dataset, valid_dataset = ConceptualCaptions(args, smalldata=True)
else:
    train_dataset, valid_dataset = ConceptualCaptions(args, smalldata=False)

# Create dataloaders
dl_opts = {**get_minimagen_dl_opts(device), 'batch_size': args.BATCH_SIZE, 'num_workers': args.NUM_WORKERS}

train_dataloader = torch.utils.data.DataLoader(train_dataset, **dl_opts)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, **dl_opts)

# Get encoding dimension of the text encoder
text_embed_dim = get_encoded_dim(args.T5_NAME)

# Create Unets
if args.RESTART_DIRECTORY is None:
    if args.TESTING:
        # If testing, use tiny MinImagen
        base_params = dict(
            dim=8,
            text_embed_dim=text_embed_dim,
            cond_dim=8,
            dim_mults=(1, 2),
            num_resnet_blocks=1,
            layer_attns=(False, False),
            layer_cross_attns=(False, False),
            attend_at_middle=False
        )

        super_params = dict(
            dim=8,
            text_embed_dim=text_embed_dim,
            cond_dim=8,
            dim_mults=(1, 2),
            num_resnet_blocks=(1, 2),
            layer_attns=(False, False),
            layer_cross_attns=(False, False),
            attend_at_middle=False
        )

        unets_params = [base_params, super_params]

        imagen_params = dict(
            image_sizes=(int(args.IMG_SIDE_LEN/2), args.IMG_SIDE_LEN),
            timesteps=args.TIMESTEPS,
            cond_drop_prob=0.15,
            text_encoder_name=args.T5_NAME
        )

        imagen_params = {**get_default_args(Imagen), **imagen_params}
    elif not args.PARAMETERS:
        # If no parameters provided, use params from minimagen.Imagen.Base and minimagen.Imagen.Super built-in classes
        base_params = dict(
            dim=512,
            dim_mults=(1, 2, 3, 4),
            num_resnet_blocks=3,
            layer_attns=(False, True, True, True),
            layer_cross_attns=(False, True, True, True),
            memory_efficient=False,
            text_embed_dim=text_embed_dim
        )
        super_params = defaults = dict(
            dim=128,
            dim_mults=(1, 2, 4, 8),
            num_resnet_blocks=(2, 4, 8, 8),
            layer_attns=(False, False, False, True),
            layer_cross_attns=(False, False, False, True),
            memory_efficient=True,
            text_embed_dim=text_embed_dim
        )

        unets_params = [base_params, super_params]

        imagen_params = dict(
            image_sizes=(int(args.IMG_SIDE_LEN/2), args.IMG_SIDE_LEN),
            timesteps=args.TIMESTEPS,
            text_encoder_name=args.T5_NAME
        )
    else:
        # If parameters are provided, load them
        unets_params, imagen_params = get_model_params(args.PARAMETERS)

    # Create Unets
    unets = [Unet(**unet_params).to(device) for unet_params in unets_params]

    # Create Imagen from UNets with specified parameters
    imagen = Imagen(unets=unets, **imagen_params).to(device)
else:
    # If training is being resumed from a previous one, load all of the relevant models/info
    orig_train_dir = os.path.join(os.getcwd(), args.RESTART_DIRECTORY)
    unets_params, imagen_params = load_params(orig_train_dir)
    imagen = load_minimagen(orig_train_dir).to(device)
    unets = imagen.unets

# Fill in unspecified arguments for complete config (parameters) file
unets_params = [{**get_default_args(Unet), **i} for i in unets_params]
imagen_params = {**get_default_args(Imagen), **imagen_params}

# Add default arguments so parameters file is complete
unets_params = [{**get_default_args(Unet), **i} for i in unets_params]

# Get the size of the Imagen model in megabytes
model_size_MB = get_model_size(imagen)

# Save all training info (config files, model size, etc.)
save_training_info(args, timestamp, unets_params, imagen_params, model_size_MB, training_dir)

# Create optimizer
optimizer = optim.Adam(imagen.parameters(), lr=args.OPTIM_LR)

# Train the MinImagen instance
MinimagenTrain(timestamp, args, unets, imagen, train_dataloader, valid_dataloader, training_dir, optimizer, timeout=30)
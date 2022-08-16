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

# Command line argument parser. See `training.get_minimagen_parser()`.
parser = get_minimagen_parser()
# Add argument for when using `main.py`
parser.add_argument("-ts", "--TIMESTAMP", dest="timestamp", help="Timestamp for training directory", type=str,
                             default=None)
args = parser.parse_args()
timestamp = args.timestamp

# Get training timestamp for when running train.py as main rather than via main.py
if timestamp is None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create training directory
dir_path = f"./training_{timestamp}"
training_dir = create_directory(dir_path)

# If loading from a parameters/training directory
if args.RESTART_DIRECTORY is not None:
    args = load_restart_training_parameters(args)
elif args.PARAMETERS is not None:
    args = load_restart_training_parameters(args, justparams=True)

# If testing, lower parameter values to lower computational load and also to lower amount of data being used.
if args.TESTING:
    args = load_testing_parameters(args)
    train_dataset, valid_dataset = ConceptualCaptions(args, smalldata=True)
else:
    train_dataset, valid_dataset = ConceptualCaptions(args, smalldata=False)

# Create dataloaders
dl_opts = {**get_minimagen_dl_opts(device), 'batch_size': args.BATCH_SIZE, 'num_workers': args.NUM_WORKERS}
train_dataloader = torch.utils.data.DataLoader(train_dataset, **dl_opts)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, **dl_opts)

# Create Unets
if args.RESTART_DIRECTORY is None:
    imagen_params = dict(
        image_sizes=(int(args.IMG_SIDE_LEN / 2), args.IMG_SIDE_LEN),
        timesteps=args.TIMESTEPS,
        cond_drop_prob=0.15,
        text_encoder_name=args.T5_NAME
    )

    # If not loading a training from a checkpoint
    if args.TESTING:
        # If testing, use tiny MinImagen for low computational load
        unets_params = [get_default_args(BaseTest), get_default_args(SuperTest)]

    # Else if not loading Unet/Imagen settings from a config (parameters) folder, use defaults
    elif not args.PARAMETERS:
        # If no parameters provided, use params from minimagen.Imagen.Base and minimagen.Imagen.Super built-in classes
        unets_params = [get_default_args(Base), get_default_args(Super)]

    # Else load unet/Imagen configs from config (parameters) folder (override imagen+params)
    else:
        # If parameters are provided, load them
        unets_params, imagen_params = get_model_params(args.PARAMETERS)

    # Create Unets accoridng to unets_params
    unets = [Unet(**unet_params).to(device) for unet_params in unets_params]

    # Create Imagen from UNets with specified imagen parameters
    imagen = Imagen(unets=unets, **imagen_params).to(device)
else:
    # If training is being resumed from a previous one, load all relevant models/info (load config AND state dicts)
    orig_train_dir = os.path.join(os.getcwd(), args.RESTART_DIRECTORY)
    unets_params, imagen_params = load_params(orig_train_dir)
    imagen = load_minimagen(orig_train_dir).to(device)
    unets = imagen.unets

# Fill in unspecified arguments with defaults for complete config (parameters) file
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
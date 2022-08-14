from datetime import datetime

import torch.utils.data
from torch import optim


from minimagen.Imagen import Imagen
from minimagen.Unet import Unet, Base, Super
from minimagen.t5 import get_encoded_dim
from minimagen.training2 import MinimagenParser, ConceptualCaptions, testing_args, MinimagenDataloaderOpts, \
    create_directory, get_model_params, get_model_size, save_training_info, get_default_args, MinimagenTrain



# Get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create training directory
dir_path = f"./training_{timestamp}"
training_dir = create_directory(dir_path)

# Command line argument parser
parser = MinimagenParser()
args = parser.parse_args()

# If testing, lower parameter values for lower computational load and also lower amount of data being used.
if args.TESTING:
    args.__dict__ = testing_args(args)
    train_dataset, valid_dataset = ConceptualCaptions(args, smalldata=True)
else:
    train_dataset, valid_dataset = ConceptualCaptions(args, smalldata=False)

# Create dataloaders
dl_opts = {**MinimagenDataloaderOpts(), 'batch_size': args.BATCH_SIZE, 'num_workers': args.NUM_WORKERS}

train_dataloader = torch.utils.data.DataLoader(train_dataset, **dl_opts)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, **dl_opts)

# Get encoding dimension of the text encoder
text_embed_dim = get_encoded_dim(args.T5_NAME)

# Create Unets
if not args.PARAMETERS:

    base_params = super_params = get_default_args(Unet)
    base_params = {**base_params, **get_default_args(Base), 'text_embed_dim':text_embed_dim}
    super_params = {**super_params, **get_default_args(Super), 'text_embed_dim':text_embed_dim}

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
        image_sizes=(64, 128),
        timesteps=args.TIMESTEPS,
        cond_drop_prob=0.15,
        text_encoder_name=args.T5_NAME
    )

else:
    unets_params, imagen_params = get_model_params(args.PARAMETERS)

unets = [Unet(**unet_params).to(device) for unet_params in unets_params]

# Create Imagen from UNets with specified parameters
imagen = Imagen(unets=unets, **imagen_params).to(device)

model_size_MB = get_model_size(imagen)

save_training_info(args, timestamp, unets_params, imagen_params, model_size_MB, training_dir)

optimizer = optim.Adam(imagen.parameters(), lr=args.OPTIM_LR)

# Train
MinimagenTrain(timestamp, args, unets, imagen, train_dataloader, valid_dataloader, training_dir, optimizer, timeout=30)
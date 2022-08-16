import inspect
import json
import os
import signal
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
import io
import urllib
from typing import Literal

from tqdm import tqdm

import datasets
import PIL.Image
from einops import rearrange
import torch.utils.data
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor

from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent
from resize_right import resize

from minimagen import Unet
from minimagen.helpers import exists
from minimagen.t5 import t5_encode_text

USER_AGENT = get_datasets_user_agent()


class _Rescale:
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

        sample = _resize_image_to_square(sample, self.side_length)

        # If there was an error in the resizing, return None
        if sample is None:
            return None

        # Rescaling max push images out of [0,1] range, so have to standardize:
        sample -= sample.min()
        sample /= sample.max()
        return sample


class MinimagenCollator:

    def __init__(self, device):
        self.device = device

    def __call__(self, batch):
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
                batch[didx][tensor] = batch[didx][tensor].to(self.device)

        return torch.utils.data.dataloader.default_collate(batch)

# DEPRECATED - replaced with MinimagenCollator so that device could be passed in rather than calculated in the function
def _collate(batch):
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


def _fetch_images(batch, num_threads, timeout=None, retries=0):
    fetch_single_image_with_args = partial(_fetch_single_image, timeout=timeout, retries=retries)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch["image"] = list(executor.map(fetch_single_image_with_args, batch["image_url"]))
    return batch


def _fetch_single_image(image_url, timeout=None, retries=0):
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


def _resize_image_to_square(image: torch.tensor,
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


def get_minimagen_parser():
    """Returns parser for MinImagen training"""
    parser = ArgumentParser()
    parser.add_argument("-p", "--PARAMETERS", dest="PARAMETERS", help="Parameters directory to load Imagen from",
                        default=None, type=str)
    parser.add_argument("-n", "--NUM_WORKERS", dest="NUM_WORKERS", help="Number of workers for DataLoader", default=0,
                        type=int)
    parser.add_argument("-b", "--BATCH_SIZE", dest="BATCH_SIZE", help="Batch size", default=2, type=int)
    parser.add_argument("-mw", "--MAX_NUM_WORDS", dest="MAX_NUM_WORDS",
                        help="Maximum number of words allowed in a caption", default=64, type=int)
    parser.add_argument("-s", "--IMG_SIDE_LEN", dest="IMG_SIDE_LEN", help="Side length of square Imagen output images",
                        default=128, type=int)
    parser.add_argument("-e", "--EPOCHS", dest="EPOCHS", help="Number of training epochs", default=5, type=int)
    parser.add_argument("-t5", "--T5_NAME", dest="T5_NAME", help="Name of T5 encoder to use", default='t5_base',
                        type=str)
    parser.add_argument("-f", "--TRAIN_VALID_FRAC", dest="TRAIN_VALID_FRAC",
                        help="Fraction of dataset to use for training (vs. validation)", default=0.9, type=float)
    parser.add_argument("-t", "--TIMESTEPS", dest="TIMESTEPS", help="Number of timesteps in Diffusion process",
                        default=1000, type=int)
    parser.add_argument("-lr", "--OPTIM_LR", dest="OPTIM_LR", help="Learning rate for Adam optimizer", default=0.0001,
                        type=float)
    parser.add_argument("-ai", "--ACCUM_ITER", dest="ACCUM_ITER", help="Number of batches for gradient accumulation",
                        default=1, type=int)
    parser.add_argument("-cn", "--CHCKPT_NUM", dest="CHCKPT_NUM",
                        help="Checkpointing batch number interval", default=500, type=int)
    parser.add_argument("-vn", "--VALID_NUM", dest="VALID_NUM",
                        help="Number of validation images to use. If None, uses full amount from train/valid split",
                        default=None, type=int)
    parser.add_argument("-rd", "--RESTART_DIRECTORY", dest="RESTART_DIRECTORY",
                        help="Training directory to resume training from if restarting.", default=None, type=str)
    parser.add_argument("-test", "--TESTING", dest="TESTING", help="Whether to test with smaller dataset",
                        action='store_true')
    parser.set_defaults(TESTING=False)
    return parser


class MinimagenDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, *, encoder_name: str, max_length: int,
                 side_length: int, train: bool = True, img_transform=None):
        """
        MinImagen Dataset object

        :param hf_dataset: HuggingFace :code:`False` Dataset object (or any dictionary with a similar structure:

            {:code:`train`: {:code:`image_url`: :code:`list(<IMAGE_URLS>)`, :code:`caption`: :code:`list(<CAPTIONS>)`}

            :code:`validation`: {:code:`image_url`: :code:`list(<IMAGE_URLS>)`, :code:`caption`: :code:`list(<CAPTIONS>)`}}

            )

        :param encoder_name: Name of the T5 encoder to use.
        :param max_length: Maximum number of words allowed in a given caption.
        :param side_length: Side length to resize all images to.
        :param train: Whether train or test dataset
        :param img_transform: (optional) Transforms to be applied on a sample in addition to default :code:`ToTensor()` and
            resizing to :code:`side_length` (applied after the defaults)
        """

        split = "train" if train else "validation"

        self.urls = hf_dataset[f"{split}"]['image_url']
        self.captions = hf_dataset[f"{split}"]['caption']

        if img_transform is None:
            self.img_transform = Compose([ToTensor(), _Rescale(side_length)])
        else:
            self.img_transform = Compose([ToTensor(), _Rescale(side_length), img_transform])
        self.encoder_name = encoder_name
        self.max_length = max_length

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = _fetch_single_image(self.urls[idx])
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


def ConceptualCaptions(args, smalldata=False, testset=False):
    """
    Load `conceptual captions dataset <https://ai.google.com/research/ConceptualCaptions/>`_

    :param args: Arguments Namespace/dictionary parsed from :func:`~.minimagen.training.get_minimagen_parser`
    :param smalldata: Whether to return a small subset of the data (for testing code)
    :param testset: Whether to return the testing set (vs training/valid)
    :return: test_dataset if :code:`testset` else (train_dataset, valid_dataset)
    """
    dset = load_dataset("conceptual_captions")
    if smalldata:
        num = 16
        vi = dset['validation']['image_url'][:num]
        vc = dset['validation']['caption'][:num]
        ti = dset['train']['image_url'][:num]
        tc = dset['train']['caption'][:num]
        dset = datasets.Dataset = {'train': {
            'image_url': ti,
            'caption': tc,
        }, 'num_rows': num,
            'validation': {
                'image_url': vi,
                'caption': vc, }, 'num_rows': num}

    if testset:
        # Torch test dataset
        test_dataset = MinimagenDataset(dset, max_length=args.MAX_NUM_WORDS, train=False, encoder_name=args.T5_NAME,
                                        side_length=args.IMG_SIDE_LEN)
        return test_dataset
    else:
        # Torch train/valid dataset
        dataset_train_valid = MinimagenDataset(dset, max_length=args.MAX_NUM_WORDS, encoder_name=args.T5_NAME,
                                               train=True,
                                               side_length=args.IMG_SIDE_LEN)

        # Split into train/valid
        train_size = int(args.TRAIN_VALID_FRAC * len(dataset_train_valid))
        valid_size = len(dataset_train_valid) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset_train_valid, [train_size, valid_size])
        if args.VALID_NUM is not None:
            valid_dataset.indices = valid_dataset.indices[:args.VALID_NUM + 1]
        return train_dataset, valid_dataset


def get_minimagen_dl_opts(device):
    """Returns dictionary of default MinImagen dataloader options"""
    return {'batch_size': 4,
            'shuffle': True,
            'num_workers': 0,
            'drop_last': True,
            'collate_fn': MinimagenCollator(device)}


class _Timeout():
    """Timeout class using ALARM signal - does not work on Windows"""

    class _Timeout(Exception): pass

    def __init__(self, sec):
        self.sec = sec

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.alarm(self.sec)

    def __exit__(self, *args):
        signal.alarm(0)  # disable alarm

    def raise_timeout(self, *args):
        raise _Timeout._Timeout()


def MinimagenTrain(timestamp, args, unets, imagen, train_dataloader, valid_dataloader, training_dir, optimizer,
                   timeout=60):
    """
    Training loop for MinImagen instance

    :param timestamp: Timestamp for training.
    :param args: Arguments Namespace/dict from argparsing :func:`.minimagen.training.get_minimagen_parser` parser.
    :param unets: List of :class:`~.minimagen.Unet.Unet`s used in the Imagen instance.
    :param imagen: :class:`~.minimagen.Imagen.Imagen` instance to train.
    :param train_dataloader: Dataloader for training.
    :param valid_dataloader: Dataloader for validation.
    :param training_dir: Training directory context manager returned from :func:`~.minimagen.training.create_directory`.
    :param optimizer: Optimizer to use for training.
    :param timeout: Amount of time to spend trying to process batch before passing on to the next batch. Does not work
        on Windows.
    :return:
    """
    def train():
        images = batch['image']
        encoding = batch['encoding']
        mask = batch['mask']

        losses = [0. for i in range(len(unets))]
        for unet_idx in range(len(unets)):
            loss = imagen(images, text_embeds=encoding, text_masks=mask, unet_number=unet_idx + 1)
            losses[unet_idx] = loss.detach()
            running_train_loss[unet_idx] += loss.detach()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(imagen.parameters(), 50)

        # Gradient accumulation optimizer step (first bool for logical short-circuiting)
        if args.ACCUM_ITER == 1 or (batch_num % args.ACCUM_ITER == 0) or (batch_num + 1 == len(train_dataloader)):
            optimizer.step()
            optimizer.zero_grad()

        # Every 10% of the way through epoch, save states in case of training failure
        if batch_num % args.CHCKPT_NUM == 0:
            with training_dir():
                with open('training_progess.txt', 'a') as f:
                    f.write(f'{"-" * 10}Checkpoint created at batch number {batch_num}{"-" * 10}\n')

            # Save temporary state dicts
            with training_dir("tmp"):
                for idx in range(len(unets)):
                    model_path = f"unet_{idx}_tmp.pth"
                    torch.save(imagen.unets[idx].state_dict(), model_path)

            # Write and batch average training loss so far
            avg_loss = [i / batch_num for i in running_train_loss]
            with training_dir():
                with open('training_progess.txt', 'a') as f:
                    f.write(f'U-Nets Avg Train Losses Epoch {epoch + 1} Batch {batch_num}: '
                            f'{[round(i.item(), 3) for i in avg_loss]}\n')
                    f.write(f'U-Nets Batch Train Losses Epoch {epoch + 1} Batch {batch_num}: '
                            f'{[round(i.item(), 3) for i in losses]}\n')

            # Compute average loss across validation batches for each unet
            running_valid_loss = [0. for i in range(len(unets))]
            imagen.train(False)

            print(f'\n{"-" * 10}Validation...{"-" * 10}')
            for vbatch in tqdm(valid_dataloader):
                if not vbatch:
                    continue

                images = vbatch['image']
                encoding = vbatch['encoding']
                mask = vbatch['mask']

                for unet_idx in range(len(unets)):
                    running_valid_loss[unet_idx] += imagen(images, text_embeds=encoding,
                                                           text_masks=mask,
                                                           unet_number=unet_idx + 1).detach()

            # Write average validation loss
            avg_loss = [i / len(valid_dataloader) for i in running_valid_loss]

            # If validation loss less than previous best, save the model weights
            for i, l in enumerate(avg_loss):
                print(f"Unet {i} avg validation loss: ", l)
                if l < best_loss[i]:
                    best_loss[i] = l
                    with training_dir("state_dicts"):
                        model_path = f"unet_{i}_state_{timestamp}.pth"
                        torch.save(imagen.unets[i].state_dict(), model_path)

            with training_dir():
                with open('training_progess.txt', 'a') as f:
                    f.write(
                        f'U-Nets Avg Valid Losses: {[round(i.item(), 3) for i in avg_loss]}\n')
                    f.write(
                        f'U-Nets Best Valid Losses: {[round(i.item(), 3) for i in best_loss]}\n\n')

    best_loss = [torch.tensor(9999999) for i in range(len(unets))]
    for epoch in range(args.EPOCHS):
        print(f'\n{"-" * 20} EPOCH {epoch + 1} {"-" * 20}')
        with training_dir():
            with open('training_progess.txt', 'a') as f:
                f.write(f'{"-" * 20} EPOCH {epoch + 1} {"-" * 20}\n')

        imagen.train(True)

        running_train_loss = [0. for i in range(len(unets))]
        print(f'\n{"-" * 10}Training...{"-" * 10}')
        for batch_num, batch in tqdm(enumerate(train_dataloader)):
            try:
                with _Timeout(timeout):
                    # If batch is empty, move on to the next one
                    if not batch:
                        continue

                    train()
            except AttributeError:
                # If batch is empty, move on to the next one
                if not batch:
                    continue

                train()
            # If batch takes longer than `timeout`, go onto the next
            except _Timeout._Timeout:
                pass
            # If the training is interrupted early, save the latest state dicts
            except Exception as e:
                # Note that training aborted
                with training_dir():
                    with open('training_progess.txt', 'a') as f:
                        f.write(
                            f'\n\nTRAINING ABORTED AT EPOCH {epoch}, BATCH NUMBER {batch_num} with exception {e}. MOST RECENT STATE '
                            f'DICTS SAVED TO ./tmp IN TRAINING FOLDER')

                # Save temporary state dicts
                with training_dir("tmp"):
                    for idx in range(len(unets)):
                        model_path = f"unet_{idx}_tmp.pth"
                        torch.save(imagen.unets[idx].state_dict(), model_path)


def load_restart_training_parameters(args, justparams=False):
    """
    Load identical command line arguments when picking up from a previous training for relevant arguments. That is,
        ensures that :code:`--MAX_NUM_WORDS`, :code:`--IMG_SIDE_LEN`, :code:`--T5_NAME`, :code:`--TIMESTEPS` command
        line arguments from :func:`~.minimagen.training.get_minimagen_parser` are all identical to the original
        training when resuming from a checkpoint.

    :param args: Arguments Namespace returned from parsing :func:`~.minimagen.training.get_minimagen_parser`.
    :param justparams: Whether loading from a parameters directory rather than a full training directory.
    """
    if justparams:
        params = args.PARAMETERS
    else:
        # Get directory from which to load relevant params
        directory = args.RESTART_DIRECTORY

        # Get file to parse
        params = os.path.join(directory, "parameters")

    file = list(filter(lambda x: x.startswith("training_"), os.listdir(params)))[0]
    with open(os.path.join(params, file), 'r') as f:
        lines = f.readlines()

    # Parse relevant args into dict
    to_keep = ["MAX_NUM_WORDS", "IMG_SIDE_LEN", "T5_NAME", "TIMESTEPS"]
    lines = list(filter(lambda x: True if True in [x.startswith(f"--{i}") for i in to_keep] else False, lines))
    d = {}
    for line in lines:
        s = line.split("=")
        try:
            d[s[0][2:]] = int(s[1][:-1])
        except:
            d[s[0][2:]] = s[1][:-1]

    # Replace relevant values in arg dict
    args.__dict__ = {**args.__dict__, **d}
    return args


def load_testing_parameters(args):
    """
    Load command line arguments that are conducive to testing training scripts (i.e. low computational load).
        In particular, the following attributes of :code:`args` are changed to the specified values:

        - BATCH_SIZE = 2

        - MAX_NUM_WORDS = 32

        - IMG_SIDE_LEN = 128

        - EPOCHS = 2

        - T5_NAME = 't5_small'

        - TRAIN_VALID_FRAC = 0.5

        - TIMESTEPS = 25

        - OPTIM_LR = 0.0001

    :param args: Arguments Namespace returned from parsing :func:`~.minimagen.training.get_minimagen_parser`.
    """
    d = dict(
            BATCH_SIZE=2,
            MAX_NUM_WORDS=32,
            IMG_SIDE_LEN=128,
            EPOCHS=2,
            T5_NAME='t5_small',
            TRAIN_VALID_FRAC=0.5,
            TIMESTEPS=25,  # Do not make less than 20
            OPTIM_LR=0.0001
        )

    # Replace relevant values in arg dict
    args.__dict__ = {**args.__dict__, **d}
    return args


def create_directory(dir_path):
    """
    Creates a training directory at the given path if it does not exist already and returns a context manager that
        allows user to temporarily enter the directory (or a subdirectory) to e.g. modify files. Also creates
        subdirectories "parameters", "state_dicts", and "tmp" under the parent directory which can be similarly
        temporarily accessed by supplying a given subdirectory name to the returned context manager as an argument.

    :param dir_path: Path of directory to create
    :return: Context manager to access created training directory/subdirectories
    """
    original_dir = os.getcwd()
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        for i in ["parameters", "state_dicts", "tmp"]:
            os.makedirs(os.path.join(dir_path, i))

    @contextmanager
    def cm(subpath=""):
        os.chdir(os.path.join(dir_path, subpath))
        yield
        os.chdir(original_dir)

    return cm


def get_model_size(imagen):
    """Returns model size in MB"""
    param_size = 0
    for param in imagen.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in imagen.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return (param_size + buffer_size) / 1024 ** 2


def save_training_info(args, timestamp, unets_params, imagen_params, model_size, training_dir):
    """
    Saves training info to training directory

    :param args: Arguments Namespace/dict from argparsing :func:`~.minimagen.training.get_minimagen_parser` parser.
    :param timestamp: Training timestamp
    :param unets_params: List of parameters of Unets to save.
    :param imagen_params: Imagen parameters to save
    :param training_dir: Context manager returned from :func:`~.minimagen.training.create_directory`
    :return:
    """
    # Save the training parameters
    with training_dir("parameters"):
        with open(f"training_parameters_{timestamp}.txt", "w") as f:
            for i in args.__dict__.keys():
                f.write(f'--{i}={getattr(args, i)}\n')

    with training_dir():
        with open('training_progess.txt', 'a') as f:
            if args.RESTART_DIRECTORY is not None:
                f.write(f"STARTED FROM CHECKPOINT {args.RESTART_DIRECTORY}\n")
            f.write(f'model size: {model_size:.3f}MB\n\n')

    # Save parameters
    with training_dir("parameters"):
        for idx, param in enumerate(unets_params):
            with open(f'unet_{idx}_params_{timestamp}.json', 'w') as f:
                json.dump(param, f, indent=4)
        with open(f'imagen_params_{timestamp}.json', 'w') as f:
            json.dump(imagen_params, f, indent=4)


def get_model_params(parameters_dir):
    """
    Returns the U-Net parameters and Imagen parameters saved in a "parameters" subdirectory of a training folder.

    :param parameters_dir: "parameters" subdirectory from which to load.
    :return: (unets_params, im_params) where unets_params is a list where the parameters index corresponds to the
        Unet number in the Imagen instance.
    """
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

    for idx, filepath in enumerate(unets_params):
        print(filepath)
        with open(os.path.join(parameters_dir, f'{filepath}'), 'r') as f:
            unets_params[idx] = json.loads(f.read())

    with open(os.path.join(parameters_dir, f'{im_params}'), 'r') as f:
        im_params = json.loads(f.read())

    return unets_params, im_params


def get_default_args(object):
    """Returns a dictionary of the default arguments of a function or class"""
    # For any subclass of Unet but not Unet itself
    if issubclass(object, Unet.Unet) and not object is Unet.Unet:
        return {**get_default_args(Unet.Unet), **object.defaults}

    signature = inspect.signature(object)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

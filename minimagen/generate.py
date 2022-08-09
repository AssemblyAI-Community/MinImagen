import json
import os
import re
from contextlib import contextmanager
from datetime import datetime

import torch

from minimagen.Unet import Unet
from minimagen.Imagen import Imagen


def _create_directory(dir_path):
    """
    Creates a directory at the given path if it does not exist already and returns a context manager that allows user
        to temporarily enter the directory.
    """
    original_dir = os.getcwd()
    img_path = os.path.join(original_dir, dir_path, "generated_images")
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    elif not len(os.listdir(img_path)) == 0:
        raise FileExistsError(f"The directory {os.path.join(original_dir, img_path)} already exists and is nonempty")

    @contextmanager
    def cm():
        os.chdir(img_path)
        yield
        os.chdir(original_dir)
    return cm


def _get_best_state_dict(unet_number, files):
    """ Gets the filename for the state_dict with lowest validation accuracy for given unet number"""
    # Filter out files not for current unet
    filt_list = list(filter(lambda x: x.startswith(f"unet_{unet_number}"), files))
    # Get validation loss of best state_dict for this unet
    min_val = min([i.split("_")[-1].split(".pth")[0] for i in filt_list])
    # Get the filename for the best state_dict for this unet
    return list(filter(lambda x: x.endswith(f"{min_val}.pth"), filt_list))[0]


def _read_params(directory, filename):
    with open(os.path.join(directory, "parameters", filename), 'r') as _file:
        return json.loads(_file.read())


def _instatiate_minimagen(directory):
    """ Instantiate an Imagen model with given parameters """
    # Files in parameters directory
    files = os.listdir(os.path.join(directory, "parameters"))

    # Filter only param files for U-Nets
    unets_params_files = sorted(list(filter(lambda x: x.startswith("unet_", ), files)),
                                key=lambda x: int(x.split("_")[1]))

    # Load U-Nets / MinImagen parameters
    unets_params = [_read_params(directory, f) for f in unets_params_files]
    imagen_params_files = _read_params(directory, list(filter(lambda x: x.startswith("imagen_"), files))[0])

    return Imagen(unets=[Unet(**params) for params in unets_params], **imagen_params_files)


def load_minimagen(directory):
    """
    Load a MinImagen instance from a training directory. Automatically chooses the highest performing state dicts.

    :param directory: MinImagen training directory as structure according to `train.py`.
    :return: MinImagen instance (ready for inference).
    """
    map_location = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    minimagen = _instatiate_minimagen(directory)

    # Filepaths for all statedicts
    files = os.listdir(os.path.join(directory, "state_dicts"))

    if files != []:
        num_unets = int(max(set([i.split("_")[1] for i in list(filter(lambda x: x.startswith("unet_"), files))]))) + 1

        # Load best state for each unet in the minimagen instance
        unet_state_dicts = [_get_best_state_dict(i, files) for i in range(num_unets)]
        for idx, file in enumerate(unet_state_dicts):
            minimagen.unets[idx].load_state_dict(torch.load(os.path.join(directory, 'state_dicts', file),
                                                            map_location=map_location))
    else:
        print(f"\n\"state_dicts\" folder in {directory} is empty, using the most recent checkpoint from \"tmp\".\n")
        files = os.listdir(os.path.join(directory, "tmp"))

        if files == []:
            raise ValueError(f"Both \"/state_dicts\" and \"/tmp\" in {directory} are empty. Train the model to acquire state dictionaries for inference. ")

        max_prct = int(max(set([i.split("_")[-2] for i in files])))
        num_unets = int(max(set([i.split("_")[1] for i in list(filter(lambda x: x.startswith("unet_"), files))]))) + 1

        for unet_num in range(num_unets):
            pth = os.path.join(directory, 'tmp', f"unet_{unet_num}_{max_prct}_percent.pth")
            minimagen.unets[unet_num].load_state_dict(torch.load(pth, map_location=map_location))

    return minimagen


def sample_and_save(minimagen: Imagen,
                    captions: list, *,
                    sample_args: dict={},
                    sequential: bool = False,
                    directory: str = None,
                    filetype: str = "png"):
    """
    Generate and save images for a list of captions using a MinImagen instance. Images are saved into a
        "generated_images" directory as "image_<CAPTION_INDEX>.<FILETYPE>"

    :param minimagen: MinImagen instance to use for sampling (i.e. generating images).
    :param captions: List of captions (strings) to generate images for.
    :param sample_args: Additional keyword arguments to pass for Imagen.sample function. Do not include texts or
        return_pil_images in this dictionary.
    :param sequential: Whether to pass captions through MinImagen sequentially rather than batched. Sequential
        processing will be slower, but it circumvents storing all images at once. Should be set to True when working
        with a large number of captions or limited memory.
    :param directory: Directory to save images to. Defaults to datetime-stamped directory if not specified.
    :param filetype: Filetype of saved images.
    :return:
    """

    if directory is None:
        directory = datetime.now().strftime("generated_images_%Y%m%d_%H%M%S")

    cm = _create_directory(directory)

    if sequential:
        for idx, elt in enumerate(captions):
            with cm():
                minimagen.sample(texts=elt, return_pil_images=True, **sample_args)[0].save(f'image_{idx}.{filetype}')
    else:
        images = minimagen.sample(texts=captions, return_pil_images=True, **sample_args)

        with cm():
            for idx, img in enumerate(images):
                img.save(f'image_{idx}.{filetype}')

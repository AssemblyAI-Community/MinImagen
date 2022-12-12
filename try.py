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


enc, msk =t5_encode_text("fish walks on the moon", "t5_small", 256)


print(enc)
print(msk)
from contextlib import contextmanager
from functools import wraps

import torch
from typing import Literal, Callable
from resize_right import resize


def cast_tuple(val, length: int = None) -> tuple:
    '''
    Casts input to a tuple. If the input is a list, converts it to a tuple. If input a single value, casts it to a
        tuple of length `length`, which is 1 if not provided.
    '''
    if isinstance(val, list):
        val = tuple(val)

    output = val if isinstance(val, tuple) else ((val,) * default(length, 1))

    if exists(length):
        assert len(output) == length

    return output


def default(val, d):
    """
    Returns the input value `val` unless it is `None`, in which case the default `d` is returned if it is a value or
        `d()` is returned if it is a callable.
    """
    if exists(val):
        return val
    return d() if callable(d) else d


def eval_decorator(fn):
    """
    Decorator for sampling from Imagen. Temporarily sets the model in evaluation mode if it was training.
    """
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


def exists(val) -> bool:
    """
    Checks to see if a value is not `None`
    """
    return val is not None


def extract(a: torch.tensor, t: torch.tensor, x_shape: torch.Size) -> torch.tensor:
    """
    Extracts values from `a` using `t` as indices

    :param a: 1D tensor of length L.
    :param t: 1D tensor of length b.
    :param x_shape: Tensor of size (b, c, h, w).
    :return: Tensor of shape (b, 1, 1, 1) that selects elements of a, using t as indices of selection.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def identity(t, *args, **kwargs):
    return t


def log(t: torch.tensor, eps: float = 1e-12) -> torch.tensor:
    """
    Calculates the natural logarithm of a torch tensor, clamping values to a minimum of `eps`.
    """
    return torch.log(t.clamp(min=eps))


def maybe(fn: Callable) -> Callable:
    """
    Returns a new function that simply applies the input function in all cases where the input is not `None`. If the
        input is `None`, `maybe` returns `None`.

    Passes through function name, docstring, etc. with [functools.wraps](https://docs.python.org/3/library/functools.html#functools.wraps)
    """

    @wraps(fn)
    def inner(x):
        if not exists(x):
            return x
        return fn(x)

    return inner


def module_device(module: torch.nn.Module) -> torch.device:
    """
    Returns the device on which a Module's parameters lie
    """
    return next(module.parameters()).device


def normalize_neg_one_to_one(img: torch.tensor) -> torch.tensor:
    """
    Normalizes an image in the range (0., 1.) to be in the range (-1., 1.). Inverse of
        :func:`.unnormalize_zero_to_one`
    """
    return img * 2 - 1


@contextmanager
def null_context(*args, **kwargs):
    """
    A placeholder null context manager that does nothing.
    """
    yield


def prob_mask_like(shape: tuple, prob: float, device: torch.device) -> torch.Tensor:
    """
    For classifier free guidance. Creates a boolean mask for given input shape and probability of `True`.

    :param shape: Shape of mask.
    :param prob: Probability of True. In interval [0., 1.].
    :param device: Device to put the mask on. Should be the same as that of the tensor which it will be used on.
    :return: The mask.
    """
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def resize_image_to(image: torch.tensor,
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
    orig_image_size = image.shape[-1]

    if orig_image_size == target_image_size:
        return image

    scale_factors = target_image_size / orig_image_size
    out = resize(image, scale_factors=scale_factors, pad_mode=pad_mode)

    if exists(clamp_range):
        out = out.clamp(*clamp_range)

    return out


def right_pad_dims_to(x: torch.tensor, t: torch.tensor) -> torch.tensor:
    """
    Pads `t` with empty dimensions to the number of dimensions `x` has. If `t` does not have fewer dimensions than `x`
        it is returned without change.
    """
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def unnormalize_zero_to_one(normed_img):
    """
    Un-normalizes an image in the range (-1., 1.) to be in the range (-1., 1.). Inverse of
        :func:`.normalize_neg_one_to_one`.
    """
    return (normed_img + 1) * 0.5

import torch
from einops import rearrange
from transformers import T5Tokenizer, T5EncoderModel

MAX_LENGTH = 256

DEFAULT_T5_NAME = 't5_small'

# Variants: https://huggingface.co/docs/transformers/model_doc/t5v1.1. 1.1 versions must be finetuned.
T5_VERSIONS = {
    't5_small': {'tokenizer': None, 'model': None, 'handle': 't5-small', 'dim': 512, 'size': .24},
    't5_base': {'tokenizer': None, 'model': None, 'handle': 't5-base', 'dim': 768, 'size': .890},
    't5_large': {'tokenizer': None, 'model': None, 'handle': 't5-large', 'dim': 1024, 'size': 2.75},
    't5_3b': {'tokenizer': None, 'model': None, 'handle': 't5-3b', 'dim': 1024, 'size': 10.6},
    't5_11b': {'tokenizer': None, 'model': None, 'handle': 't5-11b', 'dim': 1024, 'size': 42.1},
    'small1.1': {'tokenizer': None, 'model': None, 'handle': 'google/t5-v1_1-small', 'dim': 512, 'size': .3},
    'base1.1': {'tokenizer': None, 'model': None, 'handle': 'google/t5-v1_1-base', 'dim': 768, 'size': .99},
    'large1.1': {'tokenizer': None, 'model': None, 'handle': 'google/t5-v1_1-large', 'dim': 1024, 'size': 3.13},
    'xl1.1': {'tokenizer': None, 'model': None, 'handle': 'google/t5-v1_1-xl', 'dim': 2048, 'size': 11.4},
    'xxl1.1': {'tokenizer': None, 'model': None, 'handle': 'google/t5-v1_1-xxl', 'dim': 4096, 'size': 44.5},
}

# Fast tokenizers: https://huggingface.co/docs/transformers/main_classes/tokenizer
def _check_downloads(name):
    if T5_VERSIONS[name]['tokenizer'] is None:
        T5_VERSIONS[name]['tokenizer'] = T5Tokenizer.from_pretrained(T5_VERSIONS[name]['handle'])
    if T5_VERSIONS[name]['model'] is None:
        T5_VERSIONS[name]['model'] = T5EncoderModel.from_pretrained(T5_VERSIONS[name]['handle'])


def t5_encode_text(text, name: str = 't5_base', max_length=MAX_LENGTH):
    """
    Encodes a sequence of text with a T5 text encoder.

    :param text: List of text to encode.
    :param name: Name of T5 model to use. Options are:

        - :code:`'t5_small'` (~0.24 GB, 512 encoding dim),

        - :code:`'t5_base'` (~0.89 GB, 768 encoding dim),

        - :code:`'t5_large'` (~2.75 GB, 1024 encoding dim),

        - :code:`'t5_3b'` (~10.6 GB, 1024 encoding dim),

        - :code:`'t5_11b'` (~42.1 GB, 1024 encoding dim),

    :return: Returns encodings and attention mask. Element **[i,j,k]** of the final encoding corresponds to the **k**-th
        encoding component of the **j**-th token in the **i**-th input list element.
    """
    _check_downloads(name)
    tokenizer = T5_VERSIONS[name]['tokenizer']
    model = T5_VERSIONS[name]['model']

    # Move to cuda is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device)
    else:
        device = torch.device('cpu')

    # Tokenize text
    tokenized = tokenizer.batch_encode_plus(
        text,
        padding='longest',
        max_length=max_length,
        truncation=True,
        return_tensors="pt",  # Returns torch.tensor instead of python integers
    )

    input_ids = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.to(device)

    model.eval()

    # Don't need gradient - T5 frozen during Imagen training
    with torch.no_grad():
        t5_output = model(input_ids=input_ids, attention_mask=attention_mask)
        final_encoding = t5_output.last_hidden_state.detach()

    # Wherever the encoding is masked, make equal to zero
    final_encoding = final_encoding.masked_fill(~rearrange(attention_mask, '... -> ... 1').bool(), 0.)

    return final_encoding, attention_mask.bool()


def get_encoded_dim(name: str) -> int:
    """
    Gets the encoding dimensionality of a given T5 encoder.
    """
    return T5_VERSIONS[name]['dim']
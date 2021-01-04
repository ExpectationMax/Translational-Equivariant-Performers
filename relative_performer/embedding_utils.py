import numpy as np

import torch
import torch.nn as nn


def ToIntTensor(pic):
    """Transform which converts the input image into a uint8 tensor.

    Otherwise it monitors the behaviour of torchvision.transforms.ToTensor().
    """

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        img = torch.ByteTensor(
            torch.ByteStorage.from_buffer(pic.tobytes()))

    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    # put it from HWC to CHW format
    img = img.permute((2, 0, 1)).contiguous()
    return img


class LookupEmbedding(nn.Module):
    """Embedding for image channels."""

    VALUES_PER_CHANNEL = 256

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        assert out_features % in_features == 0, 'Out features needs to be a multiple of in features.'

        self.embeddings = nn.Embedding(
            in_features*self.VALUES_PER_CHANNEL, out_features//in_features)

    def forward(self, inputs):
        offsets = torch.arange(0, inputs.shape[-1]) * self.VALUES_PER_CHANNEL
        # Expand axis to match last dimension of input
        added_axes = (None, ) * (len(inputs.shape) - 1)
        offsets = offsets[added_axes]
        embeddings = self.embeddings(inputs + offsets)
        # Flatten the last two dimensions
        return embeddings.view(*(embeddings.shape[:-2] + (-1,)))


class MLPEmbedding(nn.Module):
    """Embed input using a one layer MLP."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features*4),
            nn.ReLU(),
            nn.Linear(out_features*4, out_features)
        )

    def forward(self, inputs):
        return self.net(inputs)

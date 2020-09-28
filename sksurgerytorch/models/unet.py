# From https://github.com/jvanvugt/pytorch-unet
# which is adapted from https://discuss.pytorch.org/t/unet-implementation/426

import os
import io
import logging
import glob

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

LOGGER = logging.getLogger(__name__)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=6,
        padding=False,
        batch_norm=False,
        up_mode='upconv',
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Using the default arguments will yield the exact version used
        in the original paper

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(
                    prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = \
                nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :,
            diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


class LiverSegmentationDataset(Dataset):
    """
    Liver segmentation dataset.
    """
    def __init__(self, root_dir, transform=None):
        """
        :param root_dir: Directory with all the data.
        :param transform: Transform applied to a sample.
        """
        super(LiverSegmentationDataset, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.mask_files = []

        # Each patient case is a directory in root_dir.
        # Each patient directory has a images and masks directories.
        # Iterate through the directories to get the list of files.
        sub_dirs = [f.path for f in os.scandir(root_dir) if f.is_dir()]
        if not sub_dirs:
            raise ValueError("Couldn't find sub directories")
        sub_dirs.sort()

        for sub_dir in sub_dirs:
            images_sub_dir = os.path.join(sub_dir, 'images')
            masks_sub_dir = os.path.join(sub_dir, 'masks')

            for image_file in glob.iglob(os.path.join(images_sub_dir, "*.png")):
                self.image_files.append(image_file)
            for mask_file in glob.iglob(os.path.join(masks_sub_dir, "*.png")):
                self.mask_files.append(mask_file)

    def __len__(self):
        """
        Returns the size of the dataset.

        :return: Number of images in the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, index):
        """
        Returns the index-th item {image, mask} in the dataset.

        :param index: Index of the item.
        :return: index-th item in the dataset.
        """
        if torch.is_tensor(index):
            index = index.tolist()

        image = io.imread(self.image_files[index])
        mask = io.imread(self.mask_files[index])
        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample


def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    LOGGER.info(device)

    model = UNet(n_classes=2, padding=True, up_mode='upsample').to(device)
    optim = torch.optim.Adam(model.parameters())
    dataloader = ...
    epochs = 10

    for _ in range(epochs):
        for X, y in dataloader:
            X = X.to(device)  # [N, 1, H, W]
            y = y.to(device)  # [N, H, W] with class indices (0, 1)
            prediction = model(X)  # [N, 2, H, W]
            loss = F.cross_entropy(prediction, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

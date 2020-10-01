"""
Implementation of U-Net adapted from https://github.com/jvanvugt/pytorch-unet
which is adapted from https://discuss.pytorch.org/t/unet-implementation/426
"""

import os
import sys
import logging
import datetime
import platform
import getpass
import glob
from skimage import io
from skimage import transform
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sksurgerytorch import __version__

LOGGER = logging.getLogger(__name__)

#pylint:disable=invalid-name


class UNet(nn.Module):
    """
    U-Net.
    """
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
        super().__init__()
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
    """
    U-Net convolution block.
    """
    def __init__(self, in_size, out_size, padding, batch_norm):
        """

        :param in_size: number of input channels (int).
        :param out_size: number of output channels (int).
        :param padding: if True, apply padding such that the input shape
                        is the same as the output (bool).
                        This may introduce artifacts.
        :param batch_norm: use BatchNorm after layers with an
                           activation function (bool).
        """
        super().__init__()
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
    """
    U-Net upconvolution block.
    """
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        """

        :param in_size: number of input channels (int).
        :param out_size: number of output channels (int).
        :param up_mode: one of 'upconv' or 'upsample' (str).
                       'upconv' will use transposed convolutions for
                       learned upsampling.
                       'upsample' will use bilinear upsampling.
        :param padding: if True, apply padding such that the input shape
                        is the same as the output (bool).
                        This may introduce artifacts.
        :param batch_norm: use BatchNorm after layers with an
                           activation function (bool).
        """
        super().__init__()
        if up_mode == 'upconv':
            self.up = \
                nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    @staticmethod
    def center_crop(layer, target_size):
        """
        Center-crops the filters to match target_size.

        :param layer: filters to center-crop.
        :param target_size: target size to center-crop to.
        :return: center-cropped filters.
        """
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


class SegmentationDataset(Dataset):
    """
    Segmentation dataset.
    """
    def __init__(self, root_dir, transform=None):
        """
        :param root_dir: directory with all the data.
        :param transform: transform applied to a sample.
        """
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.mask_files = []

        # Each subject is a directory in root_dir.
        # Each subject directory has a images and masks directories.
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

        :return: number of images in the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, index):
        """
        Returns the index-th item {image, mask} in the dataset.

        :param index: index of the item.
        :return: index-th item in the dataset.
        """
        if torch.is_tensor(index):
            index = index.tolist()

        image = io.imread(self.image_files[index])
        mask = io.imread(self.mask_files[index])

        # TODO: Resize image/mask to 512x512 for now.
        image = transform.resize(image, (512, 512))
        mask = transform.resize(mask, (512, 512))

        # Swap the axes to [C, H, W] format which PyTorch uses.
        image = np.transpose(image, (2, 0, 1))

        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample


def run(log_dir,
        data_dir,
        model_path,
        mode,
        save_path,
        test_path,
        epochs,
        batch_size,
        learning_rate,
        patience):
    """
    Helper function to run the U-Net model from
    the command line entry point.

    :param log_dir: directory for log files for tensorboard.
    :param data_dir: root directory of training data.
    :param model_path: file of previously saved model.
    :param mode: running mode of the model (str).
                 'train': training,
                 'test': testing.
    :param save_path: file to save model to.
    :param test_path: input image/directory to test.
    :param epochs: number of epochs.
    :param batch_size: batch size.
    :param learning_rate: learning rate for optimizer.
    :param patience: number of steps to tolerate non-improving accuracy
    """
    now = datetime.datetime.now()
    date_format = now.today().strftime("%Y-%m-%d")
    time_format = now.time().strftime("%H-%M-%S")
    logfile_name = 'unet-' \
                   + date_format \
                   + '-' \
                   + time_format \
                   + '-' \
                   + str(os.getpid()) \
                   + '.log'

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    file_handler = logging.FileHandler(logfile_name)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    username = getpass.getuser()

    LOGGER.info("Starting U-Net version: %s", __version__)
    LOGGER.info("Starting U-Net with username: %s.", username)
    LOGGER.info("Starting U-Net with platform: %s.", str(platform.uname()))
    LOGGER.info("Starting U-Net with cwd: %s.", os.getcwd())
    LOGGER.info("Starting U-Net with path: %s.", sys.path)
    LOGGER.info("Starting U-Net with save: %s.", save_path)
    LOGGER.info("Starting U-Net with test: %s.", test_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    LOGGER.info(device)

    unet = UNet(in_channels=3,
                n_classes=1,
                padding=True,
                up_mode='upsample').to(device)
    # unet = unet.float()

    if mode == 'train':
        optim = torch.optim.Adam(unet.parameters())

        train_dataset = SegmentationDataset(data_dir)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=True, num_workers=0)

        for _ in range(epochs):
            for i_batch, sample_batched in enumerate(train_dataloader):
                image_batch = sample_batched['image'].to(device)  # [N, 1, H, W]
                mask_batch = sample_batched['mask'].to(device)  # [N, H, W] with class indices (0, 1)
                image_batch = image_batch.float()
                # mask_batch = mask_batch.float()
                mask_batch = torch.reshape(mask_batch, (-1, 1, 512, 512))
                prediction = unet(image_batch)  # [N, 2, H, W]
                prediction = prediction.float()
                loss = F.binary_cross_entropy_with_logits(prediction, mask_batch)

                optim.zero_grad()
                loss.backward()
                optim.step()





    # unet = UNet(log_dir, data, working, omit, model,
    #                   learning_rate=learning_rate,
    #                   epochs=epochs,
    #                   batch_size=batch_size,
    #                   patience=patience
    #                   )

    # if save_path is not None:
    #     unet.save_model(save_path)
    #
    # if test_path is not None:
    #     if os.path.isfile(test_path):
    #         test_files = [test_path]
    #     elif os.path.isdir(test_path):
    #         test_files = ss.get_sorted_files_from_dir(test_path)
    #     else:
    #         raise ValueError("Invalid value for test parameter ")
    #
    #     for test_file in test_files:
    #
    #         img = io.imread(test_file)
    #
    #         start_time = datetime.datetime.now()
    #
    #         mask = unet.predict(img)
    #
    #         end_time = datetime.datetime.now()
    #         time_taken = (end_time - start_time).total_seconds()
    #
    #         LOGGER.info("Prediction on %s took %s seconds.",
    #                     test_file, str(time_taken))
    #
    #         # TODO: Change this to save predictions in the same folder.
    #         if os.path.isdir(prediction):
    #             io.imsave(
    #                 os.path.join(prediction, os.path.basename(test_file)),
    #                 mask)
    #         else:
    #             io.imsave(prediction, mask)

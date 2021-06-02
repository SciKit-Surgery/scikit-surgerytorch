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
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sksurgerytorch import __version__

LOGGER = logging.getLogger(__name__)

#pylint:disable=invalid-name,too-many-locals,not-callable,too-many-branches,too-many-statements,abstract-method


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
        in the original paper.

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
        """
        Forward pass.

        :param x: input tensors.
        :return: output tensors.
        """
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
        """
        Forward pass.

        :param x: input tensors.
        :return: output tensors.
        """
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
        """
        Forward pass.

        :param x: input tensors.
        :param bridge: tensors from the down path to be concatenated.
        :return: output tensors.
        """
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


class SegmentationDataset(Dataset):
    """
    Segmentation dataset.
    """
    def __init__(self, root_dir, transforms=None):
        """
        :param root_dir: directory with all the data.
        :param transforms: transforms applied to a sample.
        """
        super().__init__()
        self.root_dir = root_dir
        self.transforms = transforms
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

        # Resize image/mask to 256 x 256.
        image = transform.resize(image, (256, 256))
        mask = transform.resize(mask, (256, 256))

        # Swap the axes to [C, H, W] format which PyTorch uses.
        image = np.transpose(image, (2, 0, 1))

        sample = {'image': image, 'mask': mask}

        if self.transforms:
            sample = self.transforms(sample)

        return sample


def generate_mask(mask_pred, threshold):
    """
    Generates masks from the prediction from the network
    by setting as 1 if the pixel value >= threshold, 0 otherwise.

    :param mask_pred: prediction from the network.
    :param threshold: threshold for masking.
    :return: the mask.
    """
    mask_pred = mask_pred.clone()
    mask_pred[:, :, :, :][mask_pred[:, :, :, :] < threshold] = 0
    mask_pred[:, :, :, :][mask_pred[:, :, :, :] >= threshold] = 1.0
    return mask_pred


def run(log_dir,
        train_data_dir,
        val_data_dir,
        model_path,
        mode,
        save_path,
        test_path,
        epochs,
        batch_size,
        learning_rate):
    """
    Helper function to run the U-Net model from
    the command line entry point.

    :param log_dir: directory for log files for TensorBoard.
    :param train_data_dir: root directory of training data.
    :param val_data_dir: root directory of validation data.
    :param model_path: path to the pre-trained model.
    :param mode: running mode of the model (str).
                 'train': training,
                 'test': testing.
    :param save_path: file to save model to.
    :param test_path: input image/directory to test.
    :param epochs: number of epochs.
    :param batch_size: batch size.
    :param learning_rate: learning rate for optimiser.
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
    LOGGER.info("Starting U-Net with save path: %s.", save_path)
    LOGGER.info("Starting U-Net with test path: %s.", test_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    LOGGER.info(device)

    n_classes = 1
    unet = UNet(in_channels=3,
                n_classes=n_classes,
                padding=True,
                batch_norm=True,
                up_mode='upconv').to(device)

    if model_path is not None:
        # Load a pre-trained model. Assumes state_dict of the model was saved.
        unet.load_state_dict(torch.load(model_path))

    if mode == 'train':
        # For TensorBoard.
        writer = SummaryWriter(log_dir=log_dir)

        # Log in every logging_steps steps.
        train_logging_steps = 100
        val_logging_steps = 20

        optim = torch.optim.Adam(unet.parameters(), lr=learning_rate)

        train_dataset = SegmentationDataset(train_data_dir)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=True, num_workers=0)

        if val_data_dir is not None:
            val_dataset = SegmentationDataset(val_data_dir)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=0)

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch, epochs - 1))

            epoch_train_loss = 0
            epoch_val_loss = 0
            number_of_batches = 0

            for i_batch, sample_batched in enumerate(train_dataloader):
                image_batch = sample_batched['image'].to(device)
                mask_batch = sample_batched['mask'].to(device)
                image_batch = image_batch.float()
                mask_batch = torch.reshape(mask_batch,
                                           (-1,
                                            n_classes,
                                            mask_batch.shape[1],
                                            mask_batch.shape[2]))
                prediction = unet(image_batch)
                prediction = prediction.float()

                # For TensorBoard.
                global_step = i_batch + \
                    epoch * int(np.ceil(len(train_dataset) / batch_size))

                if global_step % train_logging_steps == 0:
                    image_grid = torchvision.utils.make_grid(image_batch)
                    writer.add_image('train/input_image', image_grid,
                                     global_step=global_step)
                    mask_grid = torchvision.utils.make_grid(mask_batch)
                    writer.add_image('train/input_mask', mask_grid,
                                     global_step=global_step)

                    # Threshold the prediction to generate mask.
                    thresholded_prediction = generate_mask(prediction, 0.5)

                    pred_mask_grid = \
                        torchvision.utils.make_grid(thresholded_prediction)
                    writer.add_image('train/pred_mask', pred_mask_grid,
                                     global_step=global_step)

                if n_classes == 1:
                    train_loss = F.binary_cross_entropy_with_logits(prediction,
                                                                    mask_batch)
                else:
                    train_loss = F.cross_entropy(prediction, mask_batch)

                # For TensorBoard.
                if global_step % train_logging_steps == 0:
                    writer.add_scalar('loss/train', train_loss,
                                      global_step=global_step)

                epoch_train_loss += train_loss
                number_of_batches += 1

                optim.zero_grad()
                train_loss.backward()
                optim.step()

            epoch_train_loss /= number_of_batches

            # For TensorBoard.
            writer.add_scalar('epoch_loss/train', epoch_train_loss,
                              global_step=epoch)

            # Validation.
            if val_data_dir is not None:
                with torch.no_grad():
                    # Set the model into evaluation mode.
                    unet.eval()

                    for i_batch, sample_batched in enumerate(val_dataloader):
                        image_batch = sample_batched['image'].to(device)
                        mask_batch = sample_batched['mask'].to(device)
                        image_batch = image_batch.float()
                        mask_batch = torch.reshape(mask_batch,
                                                   (-1,
                                                    n_classes,
                                                    mask_batch.shape[1],
                                                    mask_batch.shape[2]))
                        prediction = unet(image_batch)
                        prediction = prediction.float()

                        # For TensorBoard.
                        global_step = i_batch + \
                            epoch * int(np.ceil(len(val_dataset) / batch_size))

                        if global_step % val_logging_steps == 0:
                            image_grid = \
                                torchvision.utils.make_grid(image_batch)
                            writer.add_image('val/input_image', image_grid,
                                             global_step=global_step)
                            mask_grid = torchvision.utils.make_grid(mask_batch)
                            writer.add_image('val/input_mask', mask_grid,
                                             global_step=global_step)

                            # Threshold the prediction to generate mask.
                            thresholded_prediction = \
                                generate_mask(prediction, 0.5)

                            pred_mask_grid = \
                                torchvision.utils.make_grid(
                                    thresholded_prediction)
                            writer.add_image('val/pred_mask', pred_mask_grid,
                                             global_step=global_step)

                        if n_classes == 1:
                            val_loss = F.binary_cross_entropy_with_logits(
                                prediction,
                                mask_batch)
                        else:
                            val_loss = F.cross_entropy(prediction, mask_batch)

                        epoch_val_loss += val_loss

                    epoch_val_loss /= (i_batch + 1)

                    # For TensorBoard.
                    writer.add_scalar('epoch_loss/val', val_loss,
                                      global_step=epoch)

                # Set the model back into training mode.
                unet.train()

        if save_path is not None:
            torch.save(unet.state_dict(), save_path)

        writer.close()
    elif mode == 'test':
        if test_path is not None:
            if os.path.isfile(test_path):
                test_files = [test_path]
            elif os.path.isdir(test_path):
                filenames = os.listdir(test_path)
                filenames.sort()
                test_files = []

                for filename in filenames:
                    test_files.append(os.path.join(test_path, filename))
            else:
                raise ValueError("Invalid value for test parameter ")

            for test_file in test_files:
                image = io.imread(test_file)
                image_width = image.shape[1]
                image_height = image.shape[0]
                image = transform.resize(image, (256, 256))

                # Swap the axes to [C, H, W] format which PyTorch uses.
                image = np.transpose(image, (2, 0, 1))

                # Convert to torch.Tensor.
                image = torch.tensor(image, device=device).float()
                image = image.unsqueeze(0)

                start_time = datetime.datetime.now()

                prediction = unet(image)

                # Threshold the prediction to generate mask.
                mask = generate_mask(prediction, 0.5)

                end_time = datetime.datetime.now()
                time_taken = (end_time - start_time).total_seconds()
                LOGGER.info("Prediction on %s took %s seconds.",
                            test_file, str(time_taken))

                mask = mask.detach().squeeze(0).cpu().numpy()

                # Swap back the axes for NumPy.
                mask = np.transpose(mask, (1, 2, 0))
                mask *= 255
                mask = transform.resize(mask, (image_height, image_width),
                                        preserve_range=True)
                mask = mask.astype(np.uint8)

                # Save to a file.
                root, ext = os.path.splitext(test_file)
                io.imsave(root + "-pred" + ext, mask)

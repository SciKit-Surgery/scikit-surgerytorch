# coding=utf-8

"""
Command line entry point for U-Net script.
"""

import argparse
from sksurgerytorch import __version__
import sksurgerytorch.models.unet as unet


def main(args=None):
    """
    Entry point for U-Net script.

    Keep as little code as possible in this file, as it's hard to unit test.
    """
    parser = argparse.ArgumentParser(description='sksurgerytorch-unet')

    parser.add_argument("-l", "--log_dir",
                        required=False,
                        type=str,
                        help="Log directory for TensorBoard.")

    parser.add_argument("-d", "--train_data_dir",
                        required=False,
                        type=str,
                        help="Root directory of data to train on.")

    parser.add_argument("-val", "--val_data_dir",
                        required=False,
                        type=str,
                        help="Root directory of data to validate on.")

    parser.add_argument("-m", "--model_path",
                        required=False,
                        type=str,
                        help="Path to pre-trained model (normally .pt) "
                             "to load.")

    parser.add_argument("-md", "--mode",
                        required=False,
                        type=str,
                        help="Running mode of the model: train or test.")

    parser.add_argument("-s", "--save_path",
                        required=False,
                        type=str,
                        help="Path to save trained model (normally .pt) to.")

    parser.add_argument("-t", "--test_path",
                        required=False,
                        type=str,
                        help="Test input image/directory.")

    parser.add_argument("-e", "--epochs",
                        required=False,
                        type=int,
                        default=50,
                        help="Number of epochs.")

    parser.add_argument("-b", "--batch_size",
                        required=False,
                        type=int,
                        default=6,
                        help="Batch size.")

    parser.add_argument("-lr", "--learning_rate",
                        required=False,
                        type=float,
                        default=0.0001,
                        help="Learning rate for optimiser (Adam).")

    version_string = __version__
    friendly_version_string = version_string if version_string else 'unknown'
    parser.add_argument(
        "--version",
        action='version',
        version='sksurgerytorch-unet version ' + friendly_version_string)

    args = parser.parse_args(args)

    unet.run(args.log_dir,
             args.train_data_dir,
             args.val_data_dir,
             args.model_path,
             args.mode,
             args.save_path,
             args.test_path,
             args.epochs,
             args.batch_size,
             args.learning_rate)

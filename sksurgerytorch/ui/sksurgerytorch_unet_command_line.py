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
                        default="logs/",
                        help="Log directory for tensorboard.")

    parser.add_argument("-d", "--data_dir",
                        required=False,
                        type=str,
                        help="Root directory of data to train on.")

    parser.add_argument("-m", "--model_path",
                        required=False,
                        type=str,
                        help="Path to pre-trained model (normally .hdf5) "
                             "to load.")

    parser.add_argument("-md", "--mode",
                        required=False,
                        type=str,
                        help="Running mode of the model: train or test.")

    parser.add_argument("-s", "--save_path",
                        required=False,
                        type=str,
                        help="Path to save trained model (normally .hdf5) to.")

    parser.add_argument("-t", "--test_path",
                        required=False,
                        type=str,
                        help="Test input image/directory.")

    parser.add_argument("-e", "--epochs",
                        required=False,
                        type=int,
                        default=5,
                        help="Number of epochs.")

    parser.add_argument("-b", "--batch_size",
                        required=False,
                        type=int,
                        default=2,
                        help="Batch size.")

    parser.add_argument("-lr", "--learning_rate",
                        required=False,
                        type=float,
                        default=0.0001,
                        help="Learning rate for optimizer (Adam).")

    parser.add_argument("-pat", "--patience",
                        required=False,
                        type=int,
                        default=5,
                        help="Patience (early stopping tolerance, #steps.)")

    version_string = __version__
    friendly_version_string = version_string if version_string else 'unknown'
    parser.add_argument(
        "--version",
        action='version',
        version='sksurgerytorch-unet version ' + friendly_version_string)

    args = parser.parse_args(args)

    unet.run(args.log_dir,
             args.data_dir,
             args.model_path,
             args.mode,
             args.save_path,
             args.test_path,
             args.epochs,
             args.batch_size,
             args.learning_rate,
             args.patience)

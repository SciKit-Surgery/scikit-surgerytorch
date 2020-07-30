# coding=utf-8

""" Command line entry point for HSMNet script. """

import argparse
from sksurgerytorch import __version__
import sksurgerytorch.models.high_res_stereo as hrs

def main(args=None):
    """
    Entry point for sksurgeryhsmnet script.

    Keep as little code as possible in this file, as it's hard to unit test.
    """
    parser = argparse.ArgumentParser(description='sksurgeryhsmnet')

    parser.add_argument("-d", "--max_disp",
                        required=False,
                        type=int,
                        default=255,
                        help="Max number of disparity levels")

    parser.add_argument("-e", "--entropy_threshold",
                        required=False,
                        type=float,
                        default=-1,
                        help="Entropy threshold")

    parser.add_argument("-v", "--level",
                        required=False,
                        type=int,
                        default=1,
                        help="Model level (1 = best & slowest, \
                             3 = worst & fastest)")

    parser.add_argument("-w", "--weights",
                        required=True,
                        type=str,
                        help="Path to saved weights")

    parser.add_argument("-s", "--scale_factor",
                        required=False,
                        type=float,
                        default=0.5,
                        help="Input scale factor")

    parser.add_argument("-l", "--left_img",
                        required=True,
                        type=str,
                        help="Left input image")

    parser.add_argument("-r", "--right_img",
                        required=True,
                        type=str,
                        help="Right input image")

    parser.add_argument("-o", "--output",
                        required=True,
                        type=str,
                        help="Output file")

    version_string = __version__
    friendly_version_string = version_string if version_string else 'unknown'
    parser.add_argument(
        "--version",
        action='version',
        version='sksurgeryrgbunet version ' + friendly_version_string)

    args = parser.parse_args(args)

    hrs.run_hsmnet_model(args.max_disp,
                         args.entropy_threshold,
                         args.level,
                         args.scale_factor,
                         args.weights,
                         args.left_img,
                         args.right_img,
                         args.output)

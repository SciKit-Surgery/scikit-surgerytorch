# coding=utf-8

"""Command line processing"""


import argparse
from sksurgerytorch import __version__
from sksurgerytorch.ui.sksurgerytorch_demo import run_demo


def main(args=None):
    """Entry point for scikit-surgerytorch application"""

    parser = argparse.ArgumentParser(description='scikit-surgerytorch')

    ## ADD POSITIONAL ARGUMENTS
    parser.add_argument("x",
                        type=int,
                        help="1st number")

    parser.add_argument("y",
                        type=int,
                        help="2nd number")

    # ADD OPTINAL ARGUMENTS
    parser.add_argument("-m", "--multiply",
                        action="store_true",
                        help="Enable multiplication of inputs."
                        )

    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Enable verbose output",
                        )

    version_string = __version__
    friendly_version_string = version_string if version_string else 'unknown'
    parser.add_argument(
        "--version",
        action='version',
        version='scikit-surgerytorch version ' + friendly_version_string)

    args = parser.parse_args(args)

    run_demo(args.x, args.y, args.multiply, args.verbose)
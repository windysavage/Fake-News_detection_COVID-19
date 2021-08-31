import os
import sys
import logging
import argparse
from pathlib import Path

import pandas as pd

from utils.utils import print_environment_info

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
handlers = [stdout_handler]
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s; %(asctime)s; %(module)s:%(funcName)s:%(lineno)d; %(message)s",
    handlers=handlers)

logger = logging.getLogger(__name__)


def cli():
    print_environment_info()
    parser = argparse.ArgumentParser(
        description="Train the fake_detection model.")
    parser.add_argument("-m", "--model", type=str, default="models/fake_detection",
                        help="Path to model definition file")
    parser.add_argument("-d", "--data", type=str, default="dataset",
                        help="Path to dataset directory")
    parser.add_argument("-e", "--epochs", type=int,
                        default=20, help="Number of epochs")
    parser.add_argument("--checkpoint_interval", type=int, default=1,
                        help="Interval of epochs between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1,
                        help="Interval of epochs between evaluations on validation set")

    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    return args


def run():
    args = cli()

    # Create output directories if missing
    os.makedirs("checkpoints", exist_ok=True)

    train_df = pd.read_csv(str(Path(args.data)) + "/train.csv")
    test_df = pd.read_csv(str(Path(args.data)) + "/test.csv")


if __name__ == "__main__":
    run()

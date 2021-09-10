import os
import sys
import logging
import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import BertTokenizerFast, AutoModel

from utils.const import special_tokens, topics
from utils.utils import print_environment_info
from utils.datasets import SloganDataset
from utils.model import FinetuneBert
from utils.trainer import Trainer

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
handlers = [stdout_handler]
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s; %(asctime)s; %(module)s:%(funcName)s:%(lineno)d; %(message)s",
    handlers=handlers)

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def cli():
    print_environment_info()
    parser = argparse.ArgumentParser(
        description="Train the fake_detection model.")
    parser.add_argument("-m", "--model", type=str, default="models/fake_detection",
                        help="Path to model definition file")
    parser.add_argument("-d", "--data", type=str, default="dataset",
                        help="Path to dataset directory")
    parser.add_argument("-b", "--batch", type=int,
                        default=5, help="Batch size")
    parser.add_argument("-e", "--epochs", type=int,
                        default=1, help="Number of epochs")
    parser.add_argument("-l", "--lr", type=float,
                        default=1e-4, help="Learning rate")
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

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_df = pd.read_csv(str(Path(args.data)) + "/train.csv")
    test_df = pd.read_csv(str(Path(args.data)) + "/test.csv")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    bert = AutoModel.from_pretrained("ckiplab/bert-base-chinese")
    tokenizer.add_special_tokens(
        {'additional_special_tokens': list(special_tokens.values()) + topics})
    bert.resize_token_embeddings(len(tokenizer))

    train_ds = SloganDataset(data=train_df.head(50), tokenizer=tokenizer)
    test_ds = SloganDataset(data=test_df, tokenizer=tokenizer)

    model = FinetuneBert(bert)
    model = model.to(device)

    hparams = {
        "batch_size": args.batch,
        "epochs": args.epochs,
        "lr": args.lr
    }

    trainer = Trainer(
        hparams=hparams,
        train_ds=train_ds,
        test_ds=test_ds,
        model=model,
        device=device
    )
    trainer.train()


if __name__ == "__main__":
    run()

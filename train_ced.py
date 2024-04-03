import os
import time
import logging
import argparse
import pandas as pd

from tqdm import tqdm

from src.utils import read_file
from src.utils import read_json
from src.utils import write_file
from src.utils import write_json  

from simpletransformers.ner import NERModel
from simpletransformers.ner import NERArgs

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path",  type=str, required=True)
    parser.add_argument("--model_type",  type=str, required=True)
    parser.add_argument("--wandb"     ,  type=str, required=True)
    
    parser.add_argument("--epoch"     ,  type=int, required=True)
    parser.add_argument("--batch"     ,  type=int, required=True)
    args = parser.parse_args()

    train_data = read_json(args.train_path)

    eval_data  = train_data[-1000:]
    train_data = train_data[:-1000]

    train_data = pd.DataFrame(
        train_data, columns=["sentence_id", "words", "labels"]
    )

    eval_data = pd.DataFrame(
        eval_data, columns=["sentence_id", "words", "labels"]
    )

    # Configure the model
    model_args = NERArgs()
    model_args.train_batch_size = args.batch
    model_args.evaluate_during_training = True
    model_args.overwrite_output_dir = True
    model_args.wandb_project = args.wandb
    model_args.num_train_epochs = args.epoch

    model = NERModel(
        "bert", args.model_type, args=model_args
    )

    # Train the model
    model.train_model(train_data, eval_data=eval_data)

    # Evaluate the model
    result, model_outputs, preds_list = model.eval_model(eval_data)
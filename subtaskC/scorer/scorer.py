import pandas as pd
import logging.handlers
import argparse
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import sys
import os
import numpy as np

"""
Scoring of SEMEVAL-Task-8--subtask-C  with the metric Mean Absolute Error (MAE)
"""
logging.basicConfig(format="%(levelname)s : %(message)s", level=logging.INFO)
COLUMNS = ["id", "label"]


def check_format(file_path):
    if not os.path.exists(file_path):
        logging.error("File doesnt exists: {}".format(file_path))
        return False

    try:
        submission = pd.read_json(file_path, lines=True)[["id", "label"]]
    except Exception as e:
        logging.error("File is not a valid csv file: {}".format(file_path))
        logging.error(e)
        return False

    for column in COLUMNS:
        if submission[column].isna().any():
            logging.error("NA value in file {} in column {}".format(file_path, column))
            return False

    if not submission["label"].dtypes == "int64":
        logging.error("Unknown datatype in file {} for column label".format(file_path))

        return False

    return True


def evaluate_position_difference(actual_position, predicted_position):
    """
    Compute the absolute difference between the actual and predicted start positions.

    Args:
    - actual_position (int): Actual start position of machine-generated text.
    - predicted_position (int): Predicted start position of machine-generated text.

    Returns:
    - int: Absolute difference between the start positions.
    """
    return abs(actual_position - predicted_position)


def evaluate(pred_fpath, gold_fpath):
    """
    Evaluates the predicted classes w.r.t. a gold file.
    Metrics are: Mean Absolute Error (MAE)

    :param pred_fpath: a csv file with predictions,
    :param gold_fpath: the original annotated csv file.

    The submission of the result file should be in jsonl format.
    It should be a lines of objects:
    {
      id     -> identifier of the test sample,
      labels -> labels (0 or 1 for subtask A and from 0 to 5 for subtask B),
    }
    """

    pred_labels = pd.read_json(pred_fpath, lines=True)[["id", "label"]]
    gold_labels = pd.read_json(gold_fpath, lines=True)[["id", "label"]]

    merged_df = pred_labels.merge(gold_labels, on="id", suffixes=("_pred", "_gold"))

    # Compute the absolute difference between the actual and predicted start positions.
    out = merged_df.apply(
        lambda row: evaluate_position_difference(row["label_gold"], row["label_pred"]),
        axis=1,
    ).values
    logging.info(f"Number of samples: {len(merged_df)}")
    # Compute the mean absolute error (MAE)
    mae = np.mean(out)
    return mae


def validate_files(pred_files):
    if not check_format(pred_files):
        logging.error("Bad format for pred file {}. Cannot score.".format(pred_files))
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold_file_path",
        "-g",
        type=str,
        required=True,
        help="Paths to the CSV file with gold annotations.",
    )
    parser.add_argument(
        "--pred_file_path",
        "-p",
        type=str,
        required=True,
        help="Path to the CSV file with predictions",
    )
    args = parser.parse_args()

    pred_file_path = args.pred_file_path
    gold_file_path = args.gold_file_path

    if validate_files(pred_file_path):
        logging.info("Prediction file format is correct")
        mae = evaluate(pred_file_path, gold_file_path)
        logging.info(f"Mean Absolute Error={mae:.5f}")

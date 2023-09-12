import os
import argparse
import logging
import json
import pandas as pd
"""
This script checks whether the results format for subtask A and subtask B is correct. 
It also provides some warnings about possible errors.

The submission of the result file should be in jsonl format. 
It should be a lines of objects:
{
  id     -> identifier of the test sample,
  labels -> labels (0 or 1 for subtask A and from 0 to 5 for subtask B),
}

"""

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
COLUMNS = ['id', 'label']


def check_format(file_path):
  if not os.path.exists(file_path):
    logging.error("File doesnt exists: {}".format(file_path))
    return False
  
  try:
    submission = pd.read_json(file_path, lines=True)[['id', 'label']]
  except:
    logging.error("File is not a valid json file: {}".format(file_path))
    return False
  
  for column in COLUMNS:
    if submission[column].isna().any():
      logging.error("NA value in file {} in column {}".format(file_path, column))
      return False
  
  if not submission['label'].isin(range(0, 2)).all():
    logging.error("Unknown Label in file {}".format(file_path))
    logging.error("Unique Labels in the file are {}".format(submission['label'].unique()))
    return False
      
  return True


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--pred_files_path", "-p", nargs='+', required=True, 
    help="Path to the files you want to check.", type=str)

  args = parser.parse_args()
  logging.info("Subtask A and B. Checking files: {}".format(args.pred_files_path))
  
  for pred_file_path in args.pred_files_path:
    check_result = check_format(pred_file_path)
    result = 'Format is correct' if check_result else 'Something wrong in file format'
    logging.info("Subtask A and B. Checking file: {}. Result: {}".format(args.pred_files_path, result))

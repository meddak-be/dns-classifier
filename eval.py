"""
Date:               11/2023
Purpose:            Evaluate the classifier using the data from the tcpdump files
Example Usage:      python3 eval.py --dataset evals/ --trained_model dns_classifier_model --output suspicious.txt
"""

import argparse
import pathlib
import joblib
import pandas as pd
import os
import sys

from utils import *
from features_extractor import *


def writeResults(botList, output_file):
    with open(output_file, "w") as output:
        for bot in botList:
            output.write(bot + "\n")

def eval(output_file=None, trained_model=None, evalFile=None, botFile=None):
    global args


    # ========= LOAD ===============
    # Load the trained classifier
    classifier = joblib.load(trained_model)

    # Prepare the test data
    with open(evalFile, 'r') as test_request_file:
        test = test_request_file.readlines()
    
    with open(botFile, 'r') as botlist:
        bots = list(botlist.readlines())

    bots = [bot.rstrip('\n') for bot in bots]
    

    # ========= PARSE ===============
    d, l = parseAndAdd(test, 0)
    data = pd.DataFrame(d)
    data = extractRawData(data, l)

    combined_data = calculateFeatures(data)

    for i, val in enumerate(combined_data["req_src"]):
        if val.split(".")[0] in bots:
            combined_data.at[i, "label"] = 1 # bot

    combined_data_test = combined_data.drop(columns=['req_src', 'label'])

    # ====================================================================================================
    # ========= Proba (human+bot) ===============
    human_threshold = 0.4 
    bot_threshold = 0.6
    y_proba = classifier.predict_proba(combined_data_test)

    mixed_behavior_indices = []
    bots = []
    for i, (human_prob, bot_prob) in enumerate(y_proba):
        if human_prob > human_threshold and bot_prob < bot_threshold and human_prob < 1.0 and bot_prob > 0.0 :
            mixed_behavior_indices.append(combined_data.at[i, "req_src"])
        elif bot_prob == 1.0:
            bots.append(combined_data.at[i, "req_src"])

    bot_list = []
    for i in mixed_behavior_indices:
        message = f"{i} is detected as mixed behavior : {i in bots}"
        print(message)
        bot_list.append(message)
    for j in bots:
        message = f"{j} is detected as bot : {j in bots}"
        print(message)
        bot_list.append(message)
    
    writeResults(bot_list, output_file)




parser = argparse.ArgumentParser(description="Dataset evaluation")
parser.add_argument("--dataset", required=True, type=pathlib.Path)
parser.add_argument("--trained_model", type=pathlib.Path)
parser.add_argument("--output", required=True, type=pathlib.Path)
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")


if __name__ == "__main__":
    global args
    args = parser.parse_args()
    # if verbose -vb is set
    if args.verbose:
        print("Verbose mode on")
    
    dataset = args.dataset # TODO: use this
    if not os.path.isdir(dataset):
        print(f"The directory {dataset} does not exist.")
        sys.exit(1)
    try:
        files = [f for f in os.listdir(dataset) if os.path.isfile(os.path.join(dataset, f))]
    except OSError as e:
        print(f"An error occurred when listing files in {dataset}: {e}")
        sys.exit(1)

    # Check if there are at least two files in the directory
    if len(files) < 2:
        print(f"The directory {dataset} does not contain enough files.")
        sys.exit(1)

    bot_file = os.path.join(dataset, files[0])
    eval_file = os.path.join(dataset, files[1])

    trained_model = args.trained_model 
    output_file = args.output

    print(f"Starting evaluation with the following parameters:")
    print(f"Bot file: {bot_file}")
    print(f"Eval file: {eval_file}")
    print(f"Trained model: {trained_model}")
    print(f"Output file: {output_file}")


    eval(output_file, trained_model, evalFile=eval_file, botFile=bot_file)
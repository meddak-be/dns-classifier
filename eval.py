import argparse
import pathlib
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from train import parseAndAdd, extractFeatures, calculateFeatures
from sklearn.preprocessing import OneHotEncoder


parser = argparse.ArgumentParser(description="Dataset evaluation")
parser.add_argument("--dataset", required=True, type=pathlib.Path)
parser.add_argument("--trained_model", type=pathlib.Path)
parser.add_argument("--output", required=True, type=pathlib.Path)

def printResult(combined_data, y_pred, print_host_only=True):
    total_bots = 0
    for i, prediction in enumerate(y_pred):
        if prediction == 1:
            host = combined_data.at[i, "req_src"]
            total_bots += 1
            if print_host_only:
                print(host)
            else:
                print("[ BOT ] Detected : ", host)
    
    print("Total bots detected: ", total_bots)

def eval():
    import joblib
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    # ========= LOAD ===============
    # Load the trained SVM classifier
    classifier = joblib.load('dns_classifier_model.pkl')

    # Prepare the test data
    with open('eval1_tcpdump.txt', 'r') as test_request_file:
        test = test_request_file.readlines()
    
    with open('eval1_botlist.txt', 'r') as botlist:
        bots = list(botlist.readlines())

    bots = [bot.rstrip('\n') for bot in bots]
    

    # ========= PARSE ===============
    d, l = parseAndAdd(test, 0)
    data = pd.DataFrame(d)
    data = extractFeatures(data, l)

    combined_data = calculateFeatures(data)

    for i, val in enumerate(combined_data["req_src"]):
        if val.split(".")[0] in bots:
            combined_data.at[i, "label"] = 1
            print("added ", val, " as bot")

    # print(combined_data)
    # combined_data.to_csv('eval_data.csv', index=False)

    # Make predictions using the trained classifier
    combined_data_test = combined_data.drop(columns=['req_src', 'label'])

    # ====================================================================================================
    # ========= Proba (human+bot) ===============
    import numpy as np
    human_threshold = 0.4  # Lower boundary for human probability
    bot_threshold = 0.6
    y_proba = classifier.predict_proba(combined_data_test)

    y_pred = np.argmax(y_proba, axis=1)

    mixed_behavior_indices = []
    bots = []
    for i, (human_prob, bot_prob) in enumerate(y_proba):
        if human_prob > human_threshold and bot_prob < bot_threshold and human_prob < 1.0 and bot_prob > 0.0 :
            mixed_behavior_indices.append(combined_data.at[i, "req_src"])
        elif bot_prob == 1.0:
            bots.append(combined_data.at[i, "req_src"])
    print(mixed_behavior_indices)

    # You could mark these instances as mixed for further analysis
    for i in mixed_behavior_indices:
        print(i, " is detected as mixed behavior : ", i in bots)
    for j in bots:
        print(j, " is detected as bot", j in bots)

    #exit(0)

    # ====================================================================================================
    # ========= Predict ===============
    y_pred = classifier.predict(combined_data_test)

    # Print the accuracy score
    print("Accuracy: ", accuracy_score(combined_data["label"], y_pred))

    # Print the classification report
    print("Classification Report: \n",classification_report(combined_data["label"], y_pred))
    #exit(0)
    
    # ========= Print result ===============
    printResult(combined_data, y_pred)

    print("Mixed behavior {}:  ".format(len(mixed_behavior_indices)), ", ".join(mixed_behavior_indices))


if __name__ == "__main__":

    #args = parser.parse_args()
    eval()
import argparse
import pathlib
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from train import parseAndAdd, extractFeatures, calculateFeatures
from sklearn.preprocessing import OneHotEncoder



def printResult(combined_data, y_pred, output_file):
    global args
    total_bots = 0
    for i, prediction in enumerate(y_pred):
        if prediction == 1:
            host = combined_data.at[i, "req_src"]
            total_bots += 1
            if args.verbose:
                print("[ BOT ] Detected : ", host)
            else:
                print(host)
            with open(output_file, "a") as output:
                output.write(host + "\n")
                
    
    print("Total bots detected: ", total_bots)

def eval(output_file=None, trained_model=None, dataset=None):
    global args
    import joblib
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    # ========= LOAD ===============
    # Load the trained classifier
    classifier = joblib.load(trained_model) #'dns_classifier_model.pkl')

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
            # print("added ", val, " as bot")

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
    # print(mixed_behavior_indices)

    # You could mark these instances as mixed for further analysis
    if args.verbose:
        for i in mixed_behavior_indices:
            print(i, " is detected as mixed behavior : ", i in bots)
        for j in bots:
            print(j, " is detected as bot", j in bots)

    #exit(0)

    # ====================================================================================================
    # ========= Predict ===============
    y_pred = classifier.predict(combined_data_test)

    if args.verbose:
        # Print the accuracy score
        print("Accuracy: ", accuracy_score(combined_data["label"], y_pred))

        # Print the classification report
        print("Classification Report: \n",classification_report(combined_data["label"], y_pred))
        #exit(0)
    
    # ========= Print result ===============
    printResult(combined_data, y_pred, output_file)

    print("Mixed behavior {}:  ".format(len(mixed_behavior_indices)), ", ".join(mixed_behavior_indices))



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
    trained_model = args.trained_model # 'dns_classifier_model.pkl'
    output_file = args.output

    eval(output_file, trained_model, dataset)
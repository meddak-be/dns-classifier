import argparse
import pathlib
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from features_extractor import *
from utils import *

def preprocessing(data1, data2):
    #human = 0, bot = 1
    d, l = parseAndAdd(data1, 0) # human
    d2, l2 = parseAndAdd(data2, 1) # bot

    data =  d + d2
    labels = l+l2
        
    data = pd.DataFrame(data)

    return extractRawData(data, labels)



def svmClassifier(X_train, y_train, X_test, y_test):
    # ==================================
    # ========= SVM Classifier =========
    # ==================================
    print("\n ==== SVM - started training ==== ")

    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # === Print the evaluation metrics ===
    print(f'Accuracy: {accuracy}')

    # print(classification_report(y_test, y_pred))
    # print(confusion_matrix(y_test, y_pred))
    # ==================================
    return svm_classifier

def rfClassifier(X_train, y_train, X_test, y_test):
    # ==================================
    # ==== Random Forest Classifier ====
    # ==================================
    print("\n ==== RandomForestClassifier - started training ==== ")

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # === Print the evaluation metrics ===
    print(f'Accuracy: {accuracy}')

    # print(classification_report(y_test, y_pred))
    # print(confusion_matrix(y_test, y_pred))
    # ==================================
    return rf_classifier

def train(data1, data2):
    
    data = preprocessing(data1, data2)

    # === Feature extraction ===
    combined_data = calculateFeatures(data)

    # === Drop the req_src column as it is not needed for training ===
    combined_data = combined_data.drop(columns=['req_src'])

    # === Save the combined_data to a csv file ===
    # combined_data.to_csv('combined_data.csv', index=False)

    # === Split dataset into training and testing sets ===
    X_train, X_test, y_train, y_test = train_test_split(combined_data.drop('label', axis=1), combined_data['label'], test_size=0.4, random_state=42)


    # ==================================
    # ==== Random Forest Classifier ====
    # ==================================
    rf_classifier = rfClassifier(X_train, y_train, X_test, y_test)

    # ==================================
    # ========= SVM Classifier =========
    # ==================================
    # svm_classifier = svmClassifier(X_train, y_train, X_test, y_test)

    
    return rf_classifier




parser = argparse.ArgumentParser(description="Optional classifier training", usage="python3 train.py --webclients webclients_tcpdump.txt --bots bots_tcpdump.txt --output dns_classifier_model.pkl")
parser.add_argument("--webclients", required=True, type=pathlib.Path)
parser.add_argument("--bots", required=True, type=pathlib.Path)
parser.add_argument("--output", required=True, type=pathlib.Path)


if __name__ == "__main__":
    
    args = parser.parse_args()
    webclients_file = args.webclients
    bots_file = args.bots
    output_file = args.output

    with open(bots_file, 'r') as bot_file: # 'bots_tcpdump.txt'
        bot_dns = bot_file.readlines()

    # Read the human DNS request file
    with open(webclients_file, 'r') as human_file: # 'webclients_tcpdump.txt'
        human_dns = human_file.readlines()

    classifier = train(human_dns, bot_dns)
    
    # Save the trained classifier for later use
    import joblib
    joblib.dump(classifier, output_file) # 'dns_classifier_model.pkl')

    # Usage example: python3 train.py --webclients webclients_tcpdump.txt --bots bots_tcpdump.txt --output dns_classifier_model.pkl

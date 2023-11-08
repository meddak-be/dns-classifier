"""
Date:               11/2023
Purpose:            Train the classifier using the data from the tcpdump files
Example Usage:      python3 train.py --webclients webclients_tcpdump.txt --bots bots_tcpdump.txt --output dns_classifier_model.pkl
"""


import argparse
import pathlib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

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

def neuralNetworkClassifier(X_train, y_train, X_test, y_test):
    # ==================================
    # ========= Neural Nework Classifier =========
    # ==================================
    print("\n ==== Neural Network - started training ==== ")
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, activation='relu', solver='adam')
    mlp_classifier.fit(X_train, y_train)

    y_pred = mlp_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    #print("Neural Network Accuracy:", accuracy)
    return mlp_classifier

def kNNClassifier(X_train, y_train, X_test, y_test):
    # ==================================
    # ========= kNN Classifier =========
    # ==================================
    print("\n ==== kNN - started training ==== ")
    knn_classifier = KNeighborsClassifier(n_neighbors=5)

    knn_classifier.fit(X_train, y_train)

    y_pred = knn_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    #print("KNN Accuracy:", accuracy)
    return knn_classifier

def svmClassifier(X_train, y_train, X_test, y_test):
    # ==================================
    # ========= SVM Classifier =========
    # ==================================
    print("\n ==== SVM - started training ==== ")

    svm_classifier = SVC(kernel='rbf', gamma='scale')
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    #print(f'Accuracy: {accuracy}')
    return svm_classifier

def rfClassifier(X_train, y_train, X_test, y_test):
    # ==================================
    # ==== Random Forest Classifier ====
    # ==================================
    print("\n ==== RandomForestClassifier - started training ==== ")



    rf_classifier = RandomForestClassifier(n_estimators=100)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # === Print the evaluation metrics ===
    #print(f'Accuracy: {accuracy}')
    return rf_classifier

def decisionTreeClassifier(X_train, y_train, X_test, y_test):
    # ==================================
    # ======= Decision Tree Classifier =======
    # ==================================
    print("\n ==== Decision Tree - started training ==== ")

    dt_classifier = DecisionTreeClassifier()

    dt_classifier.fit(X_train, y_train)

    y_pred = dt_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    return dt_classifier

def train(data1, data2):
    
    data = preprocessing(data1, data2)

    # === Feature extraction ===
    combined_data = calculateFeatures(data)

    # === Drop the req_src column as it is not needed for training ===
    combined_data = combined_data.drop(columns=['req_src'])

    # === Split dataset into training and testing sets ===
    X_train, X_test, y_train, y_test = train_test_split(combined_data.drop('label', axis=1), combined_data['label'], test_size=0.4, random_state=42)

    # Possibities:
    # 1. Decision Tree
    # 2. Random Forest
    # 3. SVM
    # 4. kNN
    # 5. Neural Network
    classifier = kNNClassifier(X_train, y_train, X_test, y_test)
    
    return classifier




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

    print(f"Starting training with the following parameters:")
    print(f"Bot file: {bots_file}")
    print(f"Webclients file: {webclients_file}")
    print(f"output model: {output_file}")

    classifier = train(human_dns, bot_dns)
    
    # Save the trained classifier for later use
    import joblib
    joblib.dump(classifier, output_file) # 'dns_classifier_model.pkl')

    # Usage example: python3 train.py --webclients webclients_tcpdump.txt --bots bots_tcpdump.txt --output dns_classifier_model.pkl

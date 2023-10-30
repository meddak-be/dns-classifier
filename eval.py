import argparse
import pathlib
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from train import parseAndAdd, extractFeatures


parser = argparse.ArgumentParser(description="Dataset evaluation")
parser.add_argument("--dataset", required=True, type=pathlib.Path)
parser.add_argument("--trained_model", type=pathlib.Path)
parser.add_argument("--output", required=True, type=pathlib.Path)

def eval():
    import joblib
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    # Load the trained SVM classifier
    classifier = joblib.load('dns_classifier_model.pkl')

    # Prepare the test data
    with open('eval1_tcpdump.txt', 'r') as test_request_file:
        test = test_request_file.readlines()
    
    with open('eval1_botlist.txt', 'r') as botlist:
        bots = list(botlist.readlines())
    bots = [bot.rstrip('\n') for bot in bots]
    


    d, l = parseAndAdd(test, "")
    data = pd.DataFrame(d)
    data = extractFeatures(data, l)

    l = ["human"]*len(data)
    for i, val in enumerate(data["req_src"]):
        if val.split(".")[0] in bots:
            l[i] = "bot"
            print("added ", val, " as bot")

    exit(0)



    
    vectorizer = CountVectorizer()
    X_test = vectorizer.fit_transform(combined_test_data)
    print(X_test.shape[1]) #number of rows


    # Get all the rows that contains bot values to compare with the prediction
    bots_token_rows = []
    feature_names = vectorizer.get_feature_names_out()
    for row_index in range(X_test.shape[0]):
        row = X_test.getrow(row_index)
        # Iterate through the non-zero elements of the row
        for col_index, value in zip(row.indices, row.data):
            #print(f"({row_index}, col : {col_index}) {value}")
            token_val = feature_names[col_index]
            if 'unamur' in token_val:
                if token_val in bots:
                    bots_token_rows.append(row_index)


    # Make predictions using the trained classifier
    y_pred = classifier.predict(X_test)
    print(y_pred)
    

if __name__ == "__main__":

    #args = parser.parse_args()
    eval()
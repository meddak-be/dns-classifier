import argparse
import pathlib
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
import re


parser = argparse.ArgumentParser(description="Optional classifier training")
parser.add_argument("--webclients", required=True, type=pathlib.Path)
parser.add_argument("--bots", required=True, type=pathlib.Path)
parser.add_argument("--output", required=True, type=pathlib.Path)

def preprocessing():
    with open('bots_tcpdump.txt', 'r') as bot_file:
        bot_dns = bot_file.readlines()

    # Read the human DNS request file
    with open('webclients_tcpdump.txt', 'r') as human_file:
        human_dns = human_file.readlines()
        
    labels = []
    temp_data = []
#13:04:06.248756 IP unamur232.46230 > one.one.one.one.domain: Flags [S], seq 2432180501, win 42340, options [mss 1460,sackOK,TS val 3460327790 ecr 0,nop,wscale 9], length 0

    temp = {} #stores request
    for entry in human_dns:
        if '> one.one.one.one.domain' in entry:
            if "Flags" in entry:
                pass
            else:
                trans_port, trans_id = entry.split()[2].split(".")[1], re.sub(r'[^0-9]', '', entry.split()[5][:-1]) 
                temp[trans_port+trans_id] = entry
        elif 'one.one.one.one.domain >' in entry:
            if "Flags" in entry:
                pass
            else:
                trans_port, trans_id = entry.split()[4].split(".")[1][:-1], re.sub(r'[^0-9]', '', entry.split()[5])
                temp_data.append({'Request': temp[trans_port+trans_id].split(), 'Response': entry.split()})
                labels.append('human')
    
    print("nice")
    for entry in bot_dns:
        if '> one.one.one.one.domain' in entry:
            if "Flags" in entry:
                pass
            else:
                trans_port, trans_id = entry.split()[2].split(".")[1], re.sub(r'[^0-9]', '', entry.split()[5][:-1])
                temp[trans_port+trans_id] = entry
        elif 'one.one.one.one.domain >' in entry:
            if "Flags" in entry:
                pass
            else:
                trans_port, trans_id = entry.split()[4].split(".")[1][:-1], re.sub(r'[^0-9]', '', entry.split()[5])
                temp_data.append({'Request': temp[trans_port+trans_id].split(), 'Response': entry.split()})
                labels.append('bot')
        
    print(temp_data[0]['Request'])
    print(temp_data[0]['Response'])
    print(temp_data[1]['Request'])
    print(temp_data[1]['Response'])

    exit(0)
    
    data = []

    data['proto'] = 1
    data['req_ts'] = temp_data['Request'].str[0]
    data['req_src'] = temp_data['Request'].str[5]
    data['req_dest'] = temp_data['Request'].str[9].str.rstrip(':')
    data['req_type'] = temp_data['Request'].str[14]
    data['req_dom'] = temp_data['Request'].str[-1].rstrip('.').rstrip(' (30)').rstrip(' (26)')
    data['req_size'] = 1

    data['res_ts'] = temp_data['Response'].str[0]
    data['res_src'] = temp_data['Response'].str[5]
    data['res_dest'] = temp_data['Response'].str[9].str.rstrip(':')
    data['res_ips'] = temp_data['Response'].strip('()')
    data['res_size'] = 1

    df = pd.DataFrame(data)


def train():
    

    # Read the bot DNS request file


    # Combine bot and human DNS request and response temp_data
    combined_data = []
    combined_data.extend([req + '\n' + res for req, res in zip(bot_dns_requests, bot_dns_responses)])
    bot_length = len(combined_data)
    combined_data.extend([req + '\n' + res for req, res in zip(human_dns_requests, human_dns_responses)])
    human_length = len(combined_data) - bot_length
    
    # Create labels for the temp_data
    y = ['bot'] * bot_length + ['human'] * human_length
    # Create a TF-IDF vectorizer for text temp_data
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(combined_data)
    print(X[:100])
    feature_names = vectorizer.get_feature_names_out()
    print(feature_names[76440])
    # mapping between original source and vectorized temp_data in order
    # to find all the token containing "unamur"
    print(len(vectorizer.get_feature_names_out()))
    #exit(0)
    # Encode the labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    #print(list(y).count(1))
    #0 = bot, 1 = human
    
    # Split the temp_data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("started training")
    # Create an SVM classifier and train it
    classifier = SVC(kernel='linear', C=1.0)
    classifier.fit(X_train, y_train)

    # Save the trained classifier for later use
    import joblib
    joblib.dump(classifier, 'dns_classifier_model.pkl')

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    y_pred = classifier.predict(X_test)
    print(y_pred)
    print(X_test)
    exit(0)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    confusion = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)



   
if __name__ == "__main__":

    #args = parser.parse_args()
    preprocessing()

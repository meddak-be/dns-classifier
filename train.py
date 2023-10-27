import argparse
import pathlib


parser = argparse.ArgumentParser(description="Optional classifier training")
parser.add_argument("--webclients", required=True, type=pathlib.Path)
parser.add_argument("--bots", required=True, type=pathlib.Path)
parser.add_argument("--output", required=True, type=pathlib.Path)

def train():
    import pandas as pd
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder
    print("start")

    # Read the bot DNS request file
    with open('bots_tcpdump.txt', 'r') as bot_file:
        bot_dns = bot_file.readlines()

    # Read the human DNS request file
    with open('webclients_tcpdump.txt', 'r') as human_file:
        human_dns = human_file.readlines()

    human_dns_requests = []
    human_dns_responses = []
    for entry in human_dns:
        if '> one.one.one.one.domain' in entry:
            human_dns_requests.append(entry)
        elif 'one.one.one.one.domain >' in entry:
            human_dns_responses.append(entry)

    bot_dns_requests = []
    bot_dns_responses = []
    for entry in bot_dns:
        if '> one.one.one.one.domain' in entry:
            bot_dns_requests.append(entry)
        elif 'one.one.one.one.domain >' in entry:
            bot_dns_responses.append(entry)

    # Combine bot and human DNS request and response data
    combined_data = []
    combined_data.extend([req + '\n' + res for req, res in zip(bot_dns_requests, bot_dns_responses)])
    bot_length = len(combined_data)
    combined_data.extend([req + '\n' + res for req, res in zip(human_dns_requests, human_dns_responses)])
    human_length = len(combined_data) - bot_length
    
    # Create labels for the data
    y = ['bot'] * bot_length + ['human'] * human_length
    # Create a TF-IDF vectorizer for text data
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(combined_data)

    # Encode the labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("started training")
    # Create an SVM classifier and train it
    classifier = SVC(kernel='linear', C=1.0)
    classifier.fit(X_train, y_train)

    # Save the trained classifier for later use
    import joblib
    joblib.dump(classifier, 'dns_classifier_model.pkl')


    y_pred = classifier.predict(X_test)



   
if __name__ == "__main__":

    #args = parser.parse_args()
    train()

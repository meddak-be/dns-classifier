import argparse
import pathlib
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


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
    
    dns_requests = []
    dns_responses = []
    for entry in test:
        if '> one.one.one.one.domain' in entry:
            dns_requests.append(entry)
        elif 'one.one.one.one.domain >' in entry:
            dns_responses.append(entry)


    combined_test_data = [req.strip() + '\n' + res.strip() for req, res in zip(dns_requests, dns_responses)]
    
    vectorizer = TfidfVectorizer()
    X_test = vectorizer.fit_transform(combined_test_data)
    y_test = ['bot'] * len(combined_test_data) 

    # Make predictions using the trained classifier
    y_pred = classifier.predict(X_test)

    # Evaluate the classifier's performance
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    confusion = confusion_matrix(y_test, y_pred)

if __name__ == "__main__":

    #args = parser.parse_args()
    eval()
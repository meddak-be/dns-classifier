import argparse
import pathlib
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import re


parser = argparse.ArgumentParser(description="Optional classifier training")
parser.add_argument("--webclients", required=True, type=pathlib.Path)
parser.add_argument("--bots", required=True, type=pathlib.Path)
parser.add_argument("--output", required=True, type=pathlib.Path)

def tcp_to_dns(temp_tcp, entry_list):
    # TCP format
    # 11:47:58.237702 IP one.one.one.one.domain > unamur09.60934: Flags [S.], seq 4207413419, ack 2913238429, win 65535, options [mss 1452,nop,nop,sackOK,nop,wscale 10], length 0
    trans_port = entry_list[4].split(".")[1][:-1]
    trans_id = re.sub(r'[^0-9]', '', entry_list[5])

    req = temp_tcp[trans_port+trans_id].split()
    res = entry_list
    res_flags = res[6].split("[")[1].split("]")[0]
    req_flags = req[6].split("[")[1].split("]")[0]
    # P : push, R : reset, S : syn, F : fin, A : ack, . : no flag, P. : push and no flag, PA : push and ack
    
    format_req = "{} IP {} > {}: {} {} ({})" #11:47:58.460738 IP unamur026.60731 > one.one.one.one.domain: 23645+ A? usher.ttvnw.net. (33)
    if 'P' in req_flags:
        # format: 11:47:58.361458 IP unamur09.60934 > one.one.one.one.domain: Flags [P.], seq 1:77, ack 1, win 83, length 76 34858+ A? static.bookmsg.com. (74)
        # get the DNS query data
        timestamp = req[0]
        src_ip = req[2]
        dst_ip = req[4].rstrip(':')
        trans_id = req[15]
        #query_type = req[16] # A? or AAAA?
        query_type_and_domain = req[16:].rstrip(req[-1])  # A? abc.com.
        query_size = req[-1].rstrip(')').lstrip('(') 
        
        req = format_req.format(timestamp, src_ip, dst_ip, trans_id, query_type_and_domain, query_size)
    
    format_res = "{} IP {} > {}: {} {} {} ({})" # 11:47:58.476523 IP one.one.one.one.domain > unamur026.60731: 23645 2/0/0 A 23.160.0.254, A 192.108.239.254 (65)
    if 'P' in res_flags: # push in response packet -> DNS response
        # format : 11:47:58.381794 IP one.one.one.one.domain > unamur09.60934: Flags [P.], seq 1:631, ack 77, win 64, length 630 34858 37/0/0 A 88.198.209.15, A 81.81.81.81 (46)
        # get the DNS response data
        timestamp = res[0]
        src_ip = res[2]
        dst_ip = res[4].rstrip(':')
        trans_id = req[15] # same as request
        n_response = res[16] # 2/0/0
        response_type = res[17:].rstrip(req[-1]) # A 1.1.1.1, A 2.2.2.2
        response_size = res[-1].rstrip(')').lstrip('(')

        res = format_res.format(timestamp, src_ip, dst_ip, trans_id, n_response, response_type, response_size)
    return req, res


def preprocessing():
    with open('bots_tcpdump.txt', 'r') as bot_file:
        bot_dns = bot_file.readlines()

    # Read the human DNS request file
    with open('webclients_tcpdump.txt', 'r') as human_file:
        human_dns = human_file.readlines()
        
    labels = []
    data = []
#13:04:06.248756 IP unamur232.46230 > one.one.one.one.domain: Flags [S], seq 2432180501, win 42340, options [mss 1460,sackOK,TS val 3460327790 ecr 0,nop,wscale 9], length 0

    temp = {} #stores request
    for entry in human_dns:
        if '> one.one.one.one.domain' in entry:
            if "Flags" in entry:
                continue
                # parse the TCP packets of this form : 13:04:06.248756 IP unamur232.46230 > one.one.one.one.domain: Flags [S], seq 2432180501, win 42340, options [mss 1460,sackOK,TS val 3460327790 ecr 0,nop,wscale 9], length 0
                # 11:48:05.551485 IP unamur09.60936 > one.one.one.one.domain: Flags [P.], seq 1:77, ack 1, win 83, length 76 12527+ A? static.bookmsg.com. (74)
                flags = entry_list[6].split("[")[1].split("]")[0]
                if 'P' in flags:
                    trans_port, trans_id = entry_list[2].split(".")[1], re.sub(r'[^0-9]', '', entry_list[15])
                    temp_tcp[trans_port+trans_id] = entry
            else:
                trans_port, trans_id = entry.split()[2].split(".")[1], re.sub(r'[^0-9]', '', entry.split()[5][:-1]) 
                temp[trans_port+trans_id] = entry
        elif 'one.one.one.one.domain >' in entry:
            if "Flags" in entry:
                continue
                flags = entry_list[6].split("[")[1].split("]")[0]
                if 'P' in flags: # push TCP packet
                    req, res = tcp_to_dns(temp_tcp, entry_list)

                    temp_data.append({'Request': req, 'Response': res})
                    labels.append('human')
            else:
                trans_port, trans_id = entry.split()[4].split(".")[1][:-1], re.sub(r'[^0-9]', '', entry.split()[5])
                data.append({'Request': temp[trans_port+trans_id].split(), 'Response': entry.split()})
                labels.append('human')
    
    print("nice")
    for entry in bot_dns:
        if '> one.one.one.one.domain' in entry:
            if "Flags" in entry:
                continue
                # parse the TCP packets of this form : 13:04:06.248756 IP unamur232.46230 > one.one.one.one.domain: Flags [S], seq 2432180501, win 42340, options [mss 1460,sackOK,TS val 3460327790 ecr 0,nop,wscale 9], length 0
                flags = entry_list[6].split("[")[1].split("]")[0]
                if 'P' in flags: # push TCP packet
                    trans_port, trans_id = entry_list[2].split(".")[1], re.sub(r'[^0-9]', '', entry_list[15])
                    temp_tcp[trans_port+trans_id] = entry
            else:
                trans_port, trans_id = entry.split()[2].split(".")[1], re.sub(r'[^0-9]', '', entry.split()[5][:-1])
                temp[trans_port+trans_id] = entry
        elif 'one.one.one.one.domain >' in entry:
            if "Flags" in entry:
                continue
                # 11:48:05.573400 IP one.one.one.one.domain > unamur09.60936: Flags [P.], seq 1:631, ack 77, win 64, length 630 12527 37/0/0 A 78.47.199.218, A 78.47.199.206, A 159.69.161.134, A 88.198.200.20, A 78.47.181.156, A 116.202.204.10, A 159.69.161.138, A 88.198.209.13, A 168.119.25.64, A 88.198.204.166, A 88.198.136.226, A 78.47.199.202, A 168.119.25.20, A 88.198.136.228, A 159.69.163.6, A 88.198.200.36, A 88.198.209.34, A 78.47.199.204, A 94.130.197.138, A 88.198.204.164, A 94.130.197.136, A 168.119.25.62, A 168.119.25.78, A 94.130.197.142, A 88.198.200.22, A 88.198.136.234, A 138.201.237.88, A 159.69.167.66, A 88.198.204.168, A 78.47.199.210, A 138.201.236.216, A 88.198.209.36, A 116.202.204.12, A 94.130.197.140, A 88.198.209.15, A 168.119.25.18, A 168.119.25.66 (628)
                flags = entry_list[6].split("[")[1].split("]")[0]
                if 'P' in flags: # push TCP packet
                    req, res = tcp_to_dns(temp_tcp, entry_list)

                    temp_data.append({'Request': req, 'Response': res})
                    labels.append('human')
            else:
                trans_port, trans_id = entry.split()[4].split(".")[1][:-1], re.sub(r'[^0-9]', '', entry.split()[5])
                data.append({'Request': temp[trans_port+trans_id].split(), 'Response': entry.split()})
                labels.append('bot')
        
    data = pd.DataFrame(data)

    data['proto'] = 1
    data['req_ts'] = data['Request'].str[0]
    data['req_src'] = data['Request'].str[2]
    data['req_dest'] = data['Request'].str[4].str.rstrip(':')
    data['req_type'] = data['Request'].str[6].str.rstrip('?')
    data['req_dom'] = data['Request'].str[8]
    data['req_size'] = data['Request'].str[8].str.strip("()")

    data['res_ts'] = data['Response'].str[0]
    data['res_src'] = data['Response'].str[2]
    data['res_dest'] = data['Response'].str[4].str.rstrip(':')
    data['res_ips'] = data['Response'].str[9:-1].str.join(', ')
    data['res_size'] = data['Response'].str[-1].str.strip("()")

    selected_features = data[['req_ts', 'req_src', 'req_dest', 'req_type', 'req_dom', 'res_ts', 'res_src', 'res_dest', 'res_ips']]
    # add labels
    selected_features['labels'] = labels
    return selected_features

def train():
    
    data = preprocessing()
    encoder = OneHotEncoder(sparse_output=True) # sparse_output means that the output will be a sparse matrix
    # drop column labels
    toencode = data.drop(columns=['labels'])
    encoded_data = encoder.fit_transform(toencode)
    print(encoded_data)
    print(encoded_data.shape)
    exit(0)
    X = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['req_ts', 'req_src', 'req_dest', 'req_type', 'req_dom', 'res_ts', 'res_src', 'res_dest', 'res_ips']))
    #
    # mapping between original source and vectorized temp_data in order
    # to find all the token containing "unamur"
   # print(len(vectorizer.get_feature_names_out()))
    
    label_encoder = LabelEncoder()
    print(data[['labels']])
    y = label_encoder.fit_transform(data[['labels']])

    # Split the temp_data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(), test_size=0.2, random_state=42)

    print("started training")
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    # Create an SVM classifier and train it
    
    y_pred = rf_classifier.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Print the evaluation metrics
    print(f'Accuracy: {accuracy}')
    exit(0)
    # Save the trained classifier for later use
    import joblib
    joblib.dump(classifier, 'dns_classifier_model.pkl')


    #metrics
    
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
    train()

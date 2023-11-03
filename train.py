import argparse
import pathlib
import pandas as pd
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
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

def requestFrequency(data):
    """
    Mean of the frequency of requests per hour for each given IP
    """  

    data['req_ts'] = pd.to_datetime(data['req_ts'], format="%H:%M:%S.%f") #datetime conversion
    data['hour'] = data['req_ts'].dt.hour

    #count the number of requests
    hourly_request_count = data.groupby(['req_src', 'label', 'hour']).size().reset_index(name='request_count')
    mean_request_count = hourly_request_count.groupby(['req_src', 'label'])['request_count'].mean().reset_index(name='mean_request_count')
    return mean_request_count

def tldRequest(data):
    """
    Top-Level Domain (TLD) Requested
    """
    # TODO
    data['tld'] = data['req_dom'].str.extract(r'(\.[a-zA-Z]{2,})$')

    print(data[['req_dom', 'tld']])


def uniqueDomainRequest(data):
    """
    Number of Unique Domains Requested per hour for each given host
    """

    data['req_ts'] = pd.to_datetime(data['req_ts'], format="%H:%M:%S.%f")  #datetime conversion

    data['hour'] = data['req_ts'].dt.floor('H')  # Use floor to align timestamps to the start of the hour

    # get unique domains per host
    unique_domains_per_host = data.groupby(['req_src', 'label', 'hour'])['req_dom'].nunique().reset_index(name='unique_domains_count')
    mean_unique_domains_per_host = unique_domains_per_host.groupby(['req_src', 'label'])['unique_domains_count'].mean().reset_index(name='mean_unique_domains_count')
    return mean_unique_domains_per_host


def domainLength(data):
    """
    Calculate the mean domain length per host
    """

    data['domain_length'] = data['req_dom'].str.len()
    mean_domain_length_per_host = data.groupby(['req_src', 'label'])['domain_length'].mean().reset_index(name='mean_domain_length')

    return mean_domain_length_per_host

def intervalBetweenRequests(data):
    # TODO
    pass
    
def requestSize(data):
    """
    Calculate the mean request size per host
    """
    # numeric format conversion
    data['req_size'] = pd.to_numeric(data['req_size'], errors='coerce')  # 'coerce' to handle non-numeric values as NaN
    # mean calculation
    mean_request_size_per_host = data.groupby(['req_src', 'label'])['req_size'].mean().reset_index(name='mean_request_size')
    return mean_request_size_per_host

def queryType(data):
    """
    Return the most frequent query type per host
    """
    dns_query_type_mapping = {
        'A': 1,
        'AAAA': 2,
        'MX': 3,
        'CNAME': 4,
        'PTR': 5,
    }
    grouped_data = data.groupby(['req_src', 'label'])

    def most_frequent_query_type(group):
        query_type_counts = group['req_type'].value_counts()
        most_frequent_query = query_type_counts.idxmax()

        return dns_query_type_mapping.get(most_frequent_query, 0)

    # Apply the function to each group to get the most frequent query type per host
    most_frequent_query_per_host = grouped_data.apply(most_frequent_query_type).reset_index(name='most_frequent_query')

    return most_frequent_query_per_host

def parseAndAdd(data, label):
    dataList = []
    labelList = []
    temp = {} #stores request
    for entry in data:
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
                try:
                    dataList.append({'Request': temp[trans_port+trans_id].split(), 'Response': entry.split()})
                    labelList.append(label)
                except KeyError:
                    # TODO handle response without request
                    continue

    return dataList, labelList

def extractFeatures(data, labels):
    #pd.set_option('display.max_colwidth', None)

    data['proto'] = 1
    data['req_ts'] = data['Request'].str[0]
    sp = data['Request'].str[2].str.split('.', expand=True)
    data['req_src'],  data['req_port'] = sp[0], sp[1]
    data['req_dest'] = data['Request'].str[4].str.rstrip(':')
    data['req_type'] = data['Request'].str[6].str.rstrip('?')
    data['req_dom'] = data['Request'].str[7].str.rstrip('.')
    data['req_size'] = data['Request'].str[8].str.strip("()")

    data['res_ts'] = data['Response'].str[0]
    data['res_src'] = data['Response'].str[2]
    data['res_dest'] = data['Response'].str[4].str.rstrip(':')
    data['res_ips'] = data['Response'].str[9:-1].str.join(', ')
    data['res_size'] = data['Response'].str[-1].str.strip("()")

    selected_features = data[['req_ts', 'req_src', 'req_dest', 'req_type', 'req_dom', 'req_size', 'res_ts', 'res_src', 'res_dest', 'res_ips']]
    # add labels
    selected_features['label'] = labels

    return selected_features

def preprocessing(data1, data2):
    #human = 0, bot = 1
    d, l = parseAndAdd(data1, 0)
    d2, l2 = parseAndAdd(data2, 1)

    data =  d + d2
    labels = l+l2
        
    data = pd.DataFrame(data)

    return extractFeatures(data, labels)

def calculateFeatures(data):
    request_frequency = requestFrequency(data)
    unique_domain = uniqueDomainRequest(data)
    domain_length = domainLength(data)
    request_size = requestSize(data)
    query_type = queryType(data)

    dataframes = [request_frequency, unique_domain, domain_length, request_size, query_type]

    combined_data = dataframes[0]
    #pd.set_option('display.max_colwidth', None)
    for df in dataframes[1:]:
        # Merge the current dataframe with the combined_df on 'host' and 'label'
        combined_data = pd.merge(combined_data, df, on=['req_src', 'label'], how='outer')

    return combined_data

def train(data1, data2):
    
    data = preprocessing(data1, data2)

    # feature extraction
    combined_data = calculateFeatures(data)

    # drop the req_src column as it is not needed anymore
    combined_data = combined_data.drop(columns=['req_src'])
    
    combined_data.to_csv('combined_data.csv', index=False)

    # Split the temp_data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(combined_data.drop('label', axis=1), combined_data['label'], test_size=0.2, random_state=42)

    print("started training")
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    
    y_pred = rf_classifier.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)

    # Print the evaluation metrics
    print(f'Accuracy: {accuracy}')

    # Save the trained classifier for later use
    import joblib
    joblib.dump(rf_classifier, 'dns_classifier_model.pkl')


if __name__ == "__main__":
    with open('bots_tcpdump.txt', 'r') as bot_file:
        bot_dns = bot_file.readlines()

    # Read the human DNS request file
    with open('webclients_tcpdump.txt', 'r') as human_file:
        human_dns = human_file.readlines()
    #args = parser.parse_args()
    train(human_dns, bot_dns)

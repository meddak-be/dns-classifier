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


def tcpParseRequest(req)->str:
    req_flags = req[6].split("[")[1].split("]")[0]
    format_req = "{} IP {} > {}: {} {} ({})" #11:47:58.460738 IP unamur026.60731 > one.one.one.one.domain: 23645+ A? usher.ttvnw.net. (33)
    if 'P' in req_flags:
        # format: 11:47:58.361458 IP unamur09.60934 > one.one.one.one.domain: Flags [P.], seq 1:77, ack 1, win 83, length 76 34858+ A? static.bookmsg.com. (74)
        # get the DNS query data
        timestamp = req[0]
        src_ip = req[2]
        dst_ip = req[4].rstrip(':')
        trans_id = req[15]
        #query_type = req[16] # A? or AAAA?
        
        query_type_and_domain = " ".join(req[16:-1])  # A? abc.com.
        query_size = req[-1].rstrip(')').lstrip('(') 
        
        _req = format_req.format(timestamp, src_ip, dst_ip, trans_id, query_type_and_domain, query_size)
    return _req

def tcpParseResponse(res)->str:
    res_flags = res[6].split("[")[1].split("]")[0]
    format_res = "{} IP {} > {}: {} {} {} ({})" # 11:47:58.476523 IP one.one.one.one.domain > unamur026.60731: 23645 2/0/0 A 23.160.0.254, A 192.108.239.254 (65)
    if 'P' in res_flags: # push in response packet -> DNS response
        # format : 11:47:58.381794 IP one.one.one.one.domain > unamur09.60934: Flags [P.], seq 1:631, ack 77, win 64, length 630 34858 37/0/0 A 88.198.209.15, A 81.81.81.81 (46)
        # get the DNS response data
        timestamp = res[0]
        src_ip = res[2]
        dst_ip = res[4].rstrip(':')
        trans_id = res[15] # same as request
        n_response = res[16] # 2/0/0
        response_type = " ".join(res[17:-1]) # A 1.1.1.1, A 2.2.2.2
        response_size = res[-1].rstrip(')').lstrip('(')

        _res = format_res.format(timestamp, src_ip, dst_ip, trans_id, n_response, response_type, response_size)
    return _res

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
    Count Unique Top-Level Domain (TLD) Requested
    """
    data['tld'] = data['req_dom'].str.extract(r'(\.[a-zA-Z]{2,})$')

    # Sum of TLDs per host
    tld_count = data.groupby(['req_src', 'label', 'tld'])['tld'].count().reset_index(name='tld_count')

    # tld_count == 1 ( requested once only ) or if tld_count > 1 ( requested more than once ) then 0
    tld_count['tld_count'] = tld_count['tld_count'].apply(lambda x: 1 if x == 1 else 0)

    # Number of unique TLDs per host (sum of tld_count) requested once only
    tld = tld_count.groupby(['req_src', 'label'])['tld_count'].sum().reset_index(name='tld_count')
    return tld


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

def intervalBetweenRequests(data: pd.DataFrame):
    """
    The time between successive requests.
    Mean of the interval between requests per host
    """
    # Convert the req_ts column to datetime format
    data['req_ts'] = pd.to_datetime(data['req_ts'], format="%H:%M:%S.%f")  #datetime conversion
    # Sort the data by host and timestamp
    data = data.sort_values(by=['req_src', 'req_ts'])

    # Calculate the interval between requests
    data['interval'] = data.groupby(['req_src', 'label'])['req_ts'].diff(1).dt.total_seconds()

    # Replace the NaN values with 0
    data['interval'] = data['interval'].fillna(0)

    # Calculate the mean interval between requests per host
    mean_interval_between_requests_per_host = data.groupby(['req_src', 'label'])['interval'].mean().reset_index(name='mean_interval_between_requests')

    return mean_interval_between_requests_per_host
    
def requestSize(data):
    """
    Calculate the mean request size per host
    """
    # numeric format conversion
    data['req_size'] = pd.to_numeric(data['req_size'], errors='coerce')  # 'coerce' to handle non-numeric values as NaN
    # mean calculation
    mean_request_size_per_host = data.groupby(['req_src', 'label'])['req_size'].mean().reset_index(name='mean_request_size')
    return mean_request_size_per_host

def entropy(string):
    """
    Calculate the entropy of a given string
    """
    prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]
    entropy = - sum([p * np.log2(p) for p in prob])
    return entropy

def entropyDomain(data):
    """
    Calculate the entropy of the domain name per host
    """
    data['entropy'] = data['req_dom'].apply(lambda x: entropy(x))
    mean_entropy_per_host = data.groupby(['req_src', 'label'])['entropy'].mean().reset_index(name='mean_entropy')
    return mean_entropy_per_host


def numberOfRequestOfSameDomain(data):
    """
    Return the number of requests of the same domain per host
    """
    # 1. count the number of requests of the same domain  (for each host , label)
    data['req_dom_count'] = data.groupby(['req_src', 'label', 'req_dom'])['req_dom'].transform('count')

    # 2. get the mean of the number of requests of the same domain per host
    mean_req_dom_count_per_host = data.groupby(['req_src', 'label'])['req_dom_count'].mean().reset_index(name='mean_req_dom_count')

    return mean_req_dom_count_per_host
    
def subdomainCount(data):
    """
    Frequent access to numerous subdomains might indicate automated processes.
    Calculate the mean number of subdomains per host
    """
    data['subdomain_count'] = data['req_dom'].str.count('\.') -1 # example : www.google.com -> 2-1 = 1
    mean_subdomain_count_per_host = data.groupby(['req_src', 'label'])['subdomain_count'].mean().reset_index(name='mean_subdomain_count')
    return mean_subdomain_count_per_host

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
        entry_list = entry.split() # for optimization (instead of calling split() each time)
        if '> one.one.one.one.domain' in entry:
            if "Flags" in entry:
                # ==== TCP request ====
                #continue
                # ==== parse the TCP packets  ====
                flags = entry_list[6].split("[")[1].split("]")[0]
                if 'P' in flags:
                    trans_port, trans_id = entry_list[2].split(".")[1], re.sub(r'[^0-9]', '', entry_list[15])
                    #print("Req1",trans_port, trans_id, " => ", trans_port+ trans_id)
                    temp[trans_port+trans_id] = entry # TCP request 
            else:
                # ==== UDP request ====
                # ==== parse the UDP packets ====
                trans_port, trans_id = entry_list[2].split(".")[1], re.sub(r'[^0-9]', '', entry_list[5][:-1]) 
                temp[trans_port+trans_id] = entry
        elif 'one.one.one.one.domain >' in entry:
            if "Flags" in entry:
                # ==== TCP response ====
                #continue
                flags = entry_list[6].split("[")[1].split("]")[0]
                if 'P' in flags: # push TCP packet
                    trans_port, trans_id = entry_list[4].split(".")[1][:-1], re.sub(r'[^0-9]', '', entry_list[15])
                    if not ((trans_port+trans_id) in temp):
                        # print("Ignore tcp packet: id=", trans_id ," : port=", trans_port)
                        continue

                    req_str = temp[trans_port+trans_id]
                    res, req = tcpParseResponse(entry_list), tcpParseRequest(req_str.split()) # TCP response

                    dataList.append({'Request': req.split(), 'Response': res.split(), 'Protocol': 'TCP'})
                    labelList.append(label)
            else:
                # ==== UDP response ====
                trans_port, trans_id = entry_list[4].split(".")[1][:-1], re.sub(r'[^0-9]', '', entry_list[5])
                try:
                    dataList.append({'Request': temp[trans_port+trans_id].split(), 'Response': entry.split(), 'Protocol': 'UDP'})
                    labelList.append(label)
                except KeyError:
                    # TODO handle response without request
                    continue

    return dataList, labelList

def extractFeatures(data, labels):
    #pd.set_option('display.max_colwidth', None)

    # 1 for TCP, 0 for UDP 
    data['proto'] = data['Protocol'].apply(lambda x: 1 if x == 'TCP' else 0)
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

    selected_features = data[['req_ts','proto', 'req_src', 'req_dest', 'req_type', 'req_dom', 'req_size', 'res_ts', 'res_src', 'res_dest', 'res_ips']]
    selected_features = selected_features.copy()
    # add labels 
    selected_features['label'] = labels

    return selected_features

def preprocessing(data1, data2):
    #human = 0, bot = 1
    d, l = parseAndAdd(data1, 0) # human
    d2, l2 = parseAndAdd(data2, 1) # bot

    data =  d + d2
    labels = l+l2
        
    data = pd.DataFrame(data)

    return extractFeatures(data, labels)

def calculateFeatures(data: pd.DataFrame):
    
    # === Features ===
    # entropy_domain = entropyDomain(data)                # Calculate the entropy of the domain name per host
    subdomain_count = subdomainCount(data)              # Count the mean number of subdomains per host
    nbrReqSame_domain = numberOfRequestOfSameDomain(data) # Count the mean number of requests of the same domain per host (seem very relevant)

    tld_count = tldRequest(data)                        # Count Unique Top-Level Domain (TLD) Requested
    interval_btw_req = intervalBetweenRequests(data)    # mean of the interval between requests per host
    request_frequency = requestFrequency(data)          # mean of the frequency of requests per hour for each given IP
    unique_domain = uniqueDomainRequest(data)           # unique domain per hour
    domain_length = domainLength(data)                  # mean domain length per host
    request_size = requestSize(data)                    # mean request size per host
    query_type = queryType(data)                        # most frequent query type per host

    proto = data.groupby(['req_src', 'label'])['proto'].sum().reset_index(name='proto') # Protocol: 1 for TCP, 0 for UDP

    dataframes = [proto, request_frequency, unique_domain, domain_length, request_size, query_type, tld_count, interval_btw_req, nbrReqSame_domain, subdomain_count]
    
    combined_data = dataframes[0]
    #pd.set_option('display.max_colwidth', None)
    for df in dataframes[1:]:
        # Merge the current dataframe with the combined_df on 'host' and 'label'
        combined_data = pd.merge(combined_data, df, on=['req_src', 'label'], how='outer')

    return combined_data

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

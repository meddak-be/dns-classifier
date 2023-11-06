import numpy as np
import pandas as pd


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
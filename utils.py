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

def extractRawData(data, labels):
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
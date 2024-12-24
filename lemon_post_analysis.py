import concurrent.futures
import glob
import re
import pickle
import requests
from scipy.stats import norm
import datetime
import math
from collections import defaultdict
import numpy as np
from itertools import groupby
import os
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.dates as md
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from pathlib import Path
import time

    
def leak_events_statistics():
    file_path = './HistoricalDataAnalysis/route_leaks/*'
    files = os.path.join(
        file_path, "leak_events.*.csv")
    files = glob.glob(files)
    
    leakers = {}
    events = defaultdict(set)
    leakdict = defaultdict()
    dates = []
    n = 0
    for f in files:
        suffix = f.split('/')[-1].split('.')[-2]
        date = suffix[0:4] + '-'+suffix[4:6]+'-'+suffix[6:8]
        dates.append(date)
        with open(f) as filehandle:
            filecontents = filehandle.readlines()
            n = n + len(filecontents)
            for i, line in enumerate(filecontents):
                fields = line.strip('\n').split(',')
                # leak event,1680307200,187.63.11.0/24,3399+3356+6453+53013+53181+28658,PP,PP,6453
                timestamp = fields[1]
                t = str(datetime.datetime.fromtimestamp(
                    int(timestamp), datetime.timezone.utc))
                date = t.split(' ')[0]

                prefix, as_path, pair1, pair2, leaker = fields[2], fields[3], fields[4], fields[5], fields[-1].strip(
                )

                leakers[leaker] = 1
                events[date].add(leaker)

                if leaker not in leakdict:
                    leakdict[leaker] = {
                        'date': set(),
                        'examples': set(),
                    }
                leakdict[leaker]['date'].add(date)
                leakdict[leaker]['examples'].add(
                    (str(timestamp), prefix, leaker, as_path))
                

    c = 0
    for date in events:

        c = c + len(events[date])
    print('Events: ', n, c, len(leakers))

    frequentdict = {}
    ofile1 = open(
        './HistoricalDataAnalysis/route_leaks/leak_events_flapping.csv', 'w')
    ofile2 = open(
        './HistoricalDataAnalysis/route_leaks/leak_events.csv', 'w')
    for k, v in leakdict.items():

        if len(v['date']) > 1:

            frequentdict[k] = 1
            c = c + len(v['date'])
            
            ofile1.write(','.join([k]+sorted(v['date']))+'\n')

        else:
            ofile2.write(','.join(sorted(v['examples'])[0]) + '\n')

    ofile1.close()
    ofile2.close()
    container, numbers = [], []
    for date in dates:
        if date not in events: n = 0
        else: n = len(events[date])
        container.append((date, n))
    container = sorted(container)
    dates = []
    for date, n in container:
        dates.append(date)
        numbers.append(n)
    
    return dates, numbers
  
def quarantined_valleys_analysis():

    ty = 'quarantined_routes'
    file_path = './HistoricalDataAnalysis/route_leaks/*'
    files = os.path.join(
        file_path, ty+"_extraction.*.csv")
    files = glob.glob(files)

    objectdict = defaultdict(set)
    
    '''
    asn_types_dict = get_asn_types_dict()
    IXPs = load_IXPs()

    with open('./HistoricalDataAnalysis/route_leaks/quarantined_leaker_types.p', "rb") as f:
        leakertypes = pickle.load(f)
    print(len(leakertypes))

    nix = 0
    n, n1, n2, n3 = 0, 0, 0, 0
    for leaker in leakertypes:
        if leaker in IXPs:
            nix = nix + 1
        if leakertypes[leaker] == "Not Disclosed" or leakertypes[leaker] == "" or leakertypes[leaker] == 1:
            continue
        n = n + 1
        if "Educational/Research" in str(leakertypes[leaker]):
            n1 = n1 + 1
        if "NSP" in str(leakertypes[leaker]):
            n2 = n2 + 1

        if "Non-Profit" in str(leakertypes[leaker]):
            n3 = n3 + 1

    print(nix, n, n1, n2, n3)
    '''

    leakers = dict()
    tleak = defaultdict(set)
    valleysatday = defaultdict(set)
    dates = []
    #ofile = open('./HistoricalDataAnalysis/route_leaks/quarantined_valleys.res', 'w')
    ofile = open(
        './HistoricalDataAnalysis/route_leaks/remaining_valleys.res', 'w')
    for f in files:
        suffix = f.split('/')[-1].split('.')[-2]
        date = suffix[0:4] + '-'+suffix[4:6]+'-'+suffix[6:8]
        dates.append(date)
        with open(f) as filehandle:
            filecontents = filehandle.readlines()

            for i, line in enumerate(filecontents):
                fields = line.split(',')

                # 1680392310,2a0b:a907::/32,38880+7474+7473+1299+3356+33891+39392+51744+206468,PP,PP,33891
                timestamp, prefix, as_path, pair1, pair2, leaker = int(
                    fields[0]), fields[1], fields[2], fields[3], fields[4], fields[-1].strip('\n')

                hops = [k for k, g in groupby(as_path.split("+"))]
                as_path = '_'.join(hops)

                valley = re.search('\d+_'+leaker+'_\d+', as_path).group(0)

                objectdict[valley].add(date)
                tleak[pair1+pair2].add(valley)
                leakers[leaker] = 1
                valleysatday[date].add(valley)

    for k in tleak:
        print(k, len(tleak[k]))

    '''
    print('Total number of leakers: ', len(leakers))
    
    for leaker in leakers:
    	if leaker in leakertypes: continue
    	t = query_peerDB(leaker)
    	if t == None: continue
    	print(t)
    	leakertypes[leaker] = t
    
    with open('./HistoricalDataAnalysis/route_leaks/quarantined_leaker_types.p', "wb") as fp:
        pickle.dump(dict(leakertypes), fp)
    '''
    '''
    whitelist = dict()
    print('Total number of quarantined valley paths: ', len(objectdict))
    fres = list()
    for k in objectdict:
        # ofile.write(k+','+','.join(sorted(objectdict[k]))+'\n')
        # if len(objectdict[k]) < 2:
        #    continue
        dates = list()
        for t in sorted(objectdict[k]):
            y, m, d = t.split('-')
            date = datetime.date(int(y), int(m), int(d))
            dates.append(date)
        

        start_date = dates[0]
        whitelisted = False
        while start_date is not None:

            quarantine_days = datetime.timedelta(days=13)
            quarantine_date = start_date+quarantine_days

            quarantine_dates = [d for d in dates if d >=
                                start_date and d <= quarantine_date]
            # update start_date
            start_date = min(
                (d for d in dates if d > quarantine_date), default=None)
            if len(quarantine_dates) < 2:
                continue
            last_date = quarantine_dates[-1]
            # here we compute the average frequency of valley occurrence
            
            # and (dead_date - end_date).days <= 30
            if compute_route_fre(quarantine_dates) < 7 and (quarantine_date - last_date).days < 7:
                fre = compute_route_fre(quarantine_dates)
                fres.append(fre)
                whitelist[k] = 1
                whitelisted = True
                #ofile.write(k+','+','.join(sorted(objectdict[k]))+'\n')
                break
        if not whitelisted:
            ofile.write(k+','+','.join(sorted(objectdict[k]))+'\n')

    print('Number of whitelisted valley paths during the measurement time period: ', len(whitelist))
    ofile.close()
    n = 0
    for fre in fres:
    	if fre <= 5: n = n + 1
    print(n, len(fres))
    '''
    
    container, numbers = [], []
    for date in dates:
        if date not in valleysatday: n = 0
        else: n = len(valleysatday[date])
        container.append((date, n))
    container = sorted(container)
    dates = []
    for date, n in container:
        dates.append(date)
        numbers.append(n)
    
    return dates, numbers
  
def quarantined_asvalley_extraction():

    file_path = './HistoricalDataAnalysis/route_leaks/*/'
    files = os.path.join(
        file_path, "*.out")
    files = glob.glob(files)
    print(len(files))
    valleysdict = dict()
    valleysatday = defaultdict(set)
    leakers = dict()
    dates = []
    s = 0
    for f in files:
        
        suffix = f.split('/')[-1].split('.')[-2]
        date = suffix[0:4] + '-'+suffix[4:6]+'-'+suffix[6:8]
        dates.append(date)
        
        '''
        ofile = open(file_path+'/quarantined_routes_extraction.' +suffix+'.csv', 'w')
        qleakers = set()
        with open(file_path+'/quarantined_routes.'+suffix+'.csv') as filehandle:
            filecontents = filehandle.readlines()
            for i, line in enumerate(filecontents):
                fields = line.split(',')
                # quarantined routes,1681496846,105.183.224.0/19,267613+1299+174+8452+24863+37069,PP,PP,24863

                timestamp, prefix, as_path, pair1, pair2, leaker = int(
                    fields[1]), fields[2], fields[3], fields[4], fields[5], int(fields[-1].strip('\n'))
                qleakers.add(leaker)
         '''
        
        with open(f) as filehandle:
            filecontents = filehandle.readlines()
            s = s + len(filecontents)
            for i, line in enumerate(filecontents):
                fields = line.split(',')
                # quarantined routes,1681496846,105.183.224.0/19,267613+1299+174+8452+24863+37069,PP,PP,24863

                timestamp, prefix, as_path, pair1, pair2, leaker = int(
                    fields[0]), fields[1], fields[2], fields[3], fields[4], fields[-1].strip('\n')

                hops = [k for k, g in groupby(as_path.split("+"))]
                as_path = '_'.join(hops)

                valley = re.search('\d+_'+leaker+'_\d+', as_path).group(0)
                
                valleysdict[valley] = 1
                # if int(leaker) not in qleakers:
                #    continue
                # ofile.write(line)
                valleysatday[date].add(valley)
                leakers[leaker] = 1 

        # ofile.close()
    print('Total number of identified valley paths and unique valleys, leakers: ',
          s, len(valleysdict), len(leakers))
    
    container, numbers = [], []
    for date in dates:
        if date not in valleysatday: n = 0
        else: n = len(valleysatday[date])
        container.append((date, n))
    container = sorted(container)
    dates = []
    for date, n in container:
        dates.append(date)
        numbers.append(n)
    print(dates)
    print(numbers)
    return dates, numbers
  
def total_validated_paths():
    file_path = './HistoricalDataAnalysis/route_leaks/'
    files = os.path.join(file_path, "route_leak_validated.*.res")
    files = glob.glob(files)
    s = 0
    for f in files:
        with open(f) as filehandle:
            filecontents = filehandle.readlines()
            for i, line in enumerate(filecontents):
                fields = line.strip('\n').split(',')
                num = int(fields[1])
                print(num)
                s = s + num
    print(s)
    
def compute_gaussian_paras(data):
    u = np.mean(data)
    s = np.std(data)
    return u, s

def calculate_heg_time(date):
    t = datetime.datetime.fromtimestamp(date, datetime.timezone.utc)
    t = datetime.datetime.strftime(t, '%Y-%m-%dT%H:%M')
    mm = int(t.split('T')[1].split(':')[1])
    mm = int(mm/15) * 15
    if mm == 0:
        mm = '00'
    t = t.split(':')[0] + ':'+str(mm)
    return t
  
def is_abnormal(u, s, x):
    v = (x-u)/s
    if norm.cdf(v) > 0.95:  # p = 0.05 #orm.cdf(v) < 0.05
        return True
    return False
  
def check_forward_hegemony_value(date, originasn, asn, u, s):
    is_burst = False
    for timestamp in range(date+3600*24, date+3600*24+1, 3600*12):  # 1*15*60
        end_time = calculate_heg_time(timestamp)
        af = 4
        base_url = "https://ihr.iijlab.net/ihr/api/hegemony/?"
        # &timebin__gte=%s&timebin__lte=%s&format
        local_api = "originasn=%s&asn=%s&af=%s&timebin=%s&format=json"
        query_url = base_url + \
            local_api % (originasn, asn, af, end_time)
        x = 0.0
        rsp = None
        try:
            rsp = requests.get(query_url, headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
            }, timeout=10)
        except:
            return is_burst

        if rsp.status_code != 200:
            return is_burst
        rsp = rsp.json()
        if ('results' in rsp) and (len(rsp['results']) != 0):
            results = rsp['results']
            for res in results:
                if end_time in res['timebin']:
                    x = float(res['hege'])
                    break
        if not is_abnormal(u, s, x):

            is_burst = True
            break
    return is_burst
  
def lookup_local_hegemony_v3(date, originasn, asn):
    start_time = calculate_heg_time(date-50*15*60)
    end_time = calculate_heg_time(date+1*15*60)
    af = 4
    base_url = "https://ihr.iijlab.net/ihr/api/hegemony/?"
    # &timebin__gte=%s&timebin__lte=%s&format
    local_api = "originasn=%s&asn=%s&af=%s&timebin__gte=%s&timebin__lte=%s&format=json"
    query_url = base_url + \
        local_api % (originasn, asn, af, start_time, end_time)
    x = 0.0
    Hes = []
    Hes_dict = {}
    if query_url in Hes_dict:
        x, Hes = Hes_dict[query_url]
        return x, Hes
    rsp = None
    try:
        rsp = requests.get(query_url, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
        }, timeout=10)
    except:
        return x, Hes

    if rsp.status_code != 200:
        return x, Hes
    rsp = rsp.json()
    if ('results' in rsp) and (len(rsp['results']) != 0):
        results = rsp['results']
        for res in results:
            if end_time in res['timebin']:
                x = float(res['hege'])
                continue
            he = float(res['hege'])
            Hes.append(he)
    Hes_dict[query_url] = (x, Hes)
    return x, Hes
  

  def post_detection_v3(timestamp, asn):
    Hes = []
    abnormal = 'False'

    x, Hes = lookup_local_hegemony_v3(timestamp, '0', asn)
    u, s = 0.0, 0.0

    if x == 0 and len(Hes) <= 45:
        abnormal = 'True'
    if x != 0 and len(Hes) > 5:
        data = Hes[1:]
        u, s = compute_gaussian_paras(data)

        if is_abnormal(u, s, x):
            is_burst = check_forward_hegemony_value(timestamp, '0', asn, u, s)
            if is_burst:
                abnormal = 'True'

    return abnormal
  
def post_analyzer(date, asn, res, verified, status):
    # timestamp, prefix, asID, vrpIDs, as_path, label, scores
    if verified.get((date, asn)) != None:
        return
    if status.get(asn) != None and status.get(asn)[0]:
        return

    elif status.get(asn) != None and not status.get(asn)[0]:
        if post_detection_v3(date, asn) == 'False':
            status[asn] = (False, res)
        else:
            status[asn] = (True, res)
    else:
        if post_detection_v3(date, asn) == 'False':
            status[asn] = (False, res)
        else:
            status[asn] = (True, res)
    verified[(date, asn)] = 1
    
def resolve_afile(ifile, file_path):
    suffix = ifile.split('/')[-1].split('.')[-2]  # date
    f2 = open(file_path + 'leak_events.' + suffix+'.csv', 'w')
    f3 = open(file_path + 'quarantined_routes.' + suffix+'.csv', 'w')
    verified = {}
    status = {}
    with open(ifile) as filehandle:
        filecontents = filehandle.readlines()
        for i, line in enumerate(filecontents):
            fields = line.strip('\n').split(',')
            # 1675209600,2804:6248::/32,56662+34549+5511+6762+52871+269288,PP,PP,5511

            date, prefix, as_path, pair1, pair2, leaker = int(
                fields[0]), fields[1], fields[2], fields[3], fields[4], fields[-1]

            length = prefix.split('/')[1]
            hops = [k for k, g in groupby(as_path.split("+"))]

            if leaker != 'None':
                as_path = '_'.join(hops)
                valley = re.search('\d+_'+leaker+'_\d+', as_path).group(0)
                post_analyzer(date, leaker, fields, verified, status)

    for asn in status:
        if status[leaker][0]:
            f2.write('{},{} \n'.format('leak event',
                                       ','.join(map(str, status[asn][1]))))
        else:
            f3.write('{},{} \n'.format('quarantined routes',
                                       ','.join(map(str, status[asn][1]))))


def LeMon_post_analysis():
    #d = 'route_leaks/measure_202302/'
    d = ''
    ty = 'route_leak'
    file_path = './HistoricalDataAnalysis/'+d
    files = os.path.join(file_path, ty+".*.out")
    files = glob.glob(files)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for f in files:
            futures.append(executor.submit(resolve_afile, f, file_path))
        for future in concurrent.futures.as_completed(futures):
            future.result()


def LeMon_post_analysis_v2():
    #d = 'route_leaks/measure_202302/'
    d = ''
    ty = 'route_leak'
    file_path = './HistoricalDataAnalysis/'+d
    files = os.path.join(file_path, ty+".20230430.out")
    files = glob.glob(files)
    for f in files:
        resolve_afile(f, file_path)
        
def sample_hegemony_values(date, originasn, asn):
    start_time = calculate_heg_time(date-50*15*60)
    end_time = calculate_heg_time(date)
    # print(start_time, end_time)
    af = 4
    base_url = "https://ihr.iijlab.net/ihr/api/hegemony/?"
    # &timebin__gte=%s&timebin__lte=%s&format
    local_api = "originasn=%s&asn=%s&af=%s&timebin__gte=%s&timebin__lte=%s&format=json"
    query_url = base_url + \
        local_api % (originasn, asn, af, start_time, end_time)
    # print(query_url)
    rsp = None
    x = 0
    Hes = []
    try:
        rsp = requests.get(query_url, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
        }, timeout=10)
    except:
        return x, Hes
    if rsp.status_code != 200:
        return x, Hes
    rsp = rsp.json()
    if ('results' in rsp) and (len(rsp['results']) != 0):
        results = rsp['results']
        for res in results:
            if end_time in res['timebin']:
                x = float(res['hege'])
                continue
            he = float(res['hege'])
            Hes.append(he)
    return x, Hes

def test_bgpmon_leaks(ty, start_time, end_time, asn):
    report_time = start_time

    y, m, d = report_time.split('T')[0].split('-')
    hh, mm = report_time.split('T')[1].split(':')

    report_timestamp = int(calculate_unix_time(
        datetime.datetime(int(y), int(m), int(d), int(hh), int(mm))))

    start_timestamp = calculate_heg_time(int(calculate_unix_time(
        datetime.datetime(int(y), int(m), int(d), int(hh), int(mm)))))
    y, m, d = start_timestamp.split('T')[0].split('-')
    hh, mm = start_timestamp.split('T')[1].split(':')
    start_timestamp = int(calculate_unix_time(
        datetime.datetime(int(y), int(m), int(d), int(hh), int(mm))))

    for timestamp in range(start_timestamp-5*3600, start_timestamp+5*3600, 60*15):
        t = datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc)

        x, Hes = sample_hegemony_values(timestamp, '0', asn)

        u, s = 0.0, 0.0
        abnormal = 'False'
        if x == 0 and len(Hes) <= 45:  # we set 25 when the total number of samples is 30
            abnormal = 'True'
        if x != 0 and len(Hes) > 5:
            data = Hes[1:]
            u, s = compute_gaussian_paras(data)

            if is_abnormal(u, s, x):
                abnormal = 'True'
                '''
                is_burst = check_forward_hegemony_value(timestamp, '0', asn, u, s)
               
                if is_burst:
                    abnormal = 'True'
                '''
        if abnormal == 'True':
            delay = (timestamp - report_timestamp)/3600
            if x == 0:
                delay = 0
            #t = datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc)
            print(t, x, delay)
            return delay

    return None

def BGPmon_measurements():
    
    ty = "route_leaks"
    ofile = open('./HistoricalDataAnalysis/' + ty +
                 '/bgpmon_leaks.verified.csv', 'w')
    
    with open('./HistoricalDataAnalysis/'+ty+'/BGPmon_leaks.csv') as filehandle:
        filecontents = filehandle.readlines()
        for i, line in enumerate(filecontents):
            
            line = line.strip('\n')
            # 2022-10-27 08:34:34+00:00,100.20.30.0/24,8612+174+3320+4809+15399+37027+37063,37063
            fields = line.split(',')
            timestamp, attacker = fields[0], fields[-1]
            date = timestamp.split(' ')[0]
            st = timestamp.split(' ')[1].split(
                ':')[0] + ':' + timestamp.split(' ')[1].split(':')[1]
            et = str(int(timestamp.split(' ')[1].split(':')[
                     0])+1) + ':' + timestamp.split(' ')[1].split(':')[1]

            asn = int(attacker)
            start_time = date + 'T' + st
            end_time = date + 'T' + et

            

            delay = test_bgpmon_leaks(
                ty, start_time, end_time, asn)  # test_bgpmon_events
            if delay == None:
                continue
            ofile.write(line+','+str(int(delay))+'\n')
    ofile.close()
    
def plot_valley_numbers():
    
    fig, ax = plt.subplots(1)
    fig.set_size_inches(15, 7.5)
    
    x, y = quarantined_asvalley_extraction()
   
    mean = sum(y)/len(y)
    print(sum(y), mean)
    
    plt.plot(x, y, '-', linewidth=5, color='#ffc20a', label='Identified valleys')
    #ax.axhline(y=mean, linewidth=2, color='blue', linestyle='--', label='BGP hijack (mean)')
    
    x, y = quarantined_valleys_analysis()
    mean = sum(y)/len(y)
    print(sum(y), mean)

    plt.plot(x, y, '-', linewidth=5, color='#40b0a6',
             label='Quarantined valleys')
    #ax.axhline(y=mean, linewidth=2, color='red', linestyle='--', label='Confirmed hijack (mean)')
    
    
    '''
    x, y = leak_events_statistics()
    mean = sum(y)/len(y)
    print(mean)
    print(len(x))
    
    plt.plot(x, y, '-', linewidth=5, color='red', label='Captured events')
    #ax.axhline(y=mean, linewidth=2, color='blue', linestyle='--', label='BGP hijack (mean)')
    '''
    

    ax.set_xticks(x[::6])
    ax.set_xticklabels(x[::6], rotation=45)
    
    #ax.set_ylim(0, 80000)
    # ax.set_xlim(0, 47)
    # plt.xticks(np.arange(1,47,4))
    #ax.tick_params(axis='x', labelrotation=90)
    #ax.set_xlabel('Date', fontsize=30)
    ax.set_ylabel('Number(#)', fontsize=30)
    plt.legend(loc="upper right", bbox_to_anchor=(0.94, 1.0), fontsize=25)
    
    # plt.grid(linewidth=0.5)
    ax.tick_params(axis='both', labelsize=30)  # which='major'
    plt.tight_layout()

    #plt.show()
    plt.savefig('./valley_numbers.pdf', dpi=300)
    
        
def main():

    LeMon_post_analysis()
    total_validated_paths()
    #quarantined_asvalley_extraction()
    quarantined_valleys_analysis()
    
    leak_events_statistics()
    BGPmon_measurements()
    
    plot_valley_numbers()
    
    

if __name__ == "__main__":
    main()

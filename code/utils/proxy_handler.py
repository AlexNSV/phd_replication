from datetime import datetime
headers = {'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.5005.134 YaBrowser/22.7.1.828 (beta) Yowser/2.5 Safari/537.36', "Referer": "http://example.com"}
import threading
from queue import Queue
DEFAULT_TEST_URL = 'https://www.researchgate.net/'
from time import sleep
from lxml import html as lxmlhtml

import random
import requests
import pandas as pd

#loker = threading.Lock()
#loker.acquire()
#loker.release()


class ProxyTest():
    def __init__(self, proxy_ip, url, test_result, test_time, response_time=None):
        self.proxy_ip = proxy_ip
        self.url = url
        self.result = test_result
        self.test_time = test_time
        self.response_time = response_time
        
    def __repr__(self):
        return f'Proxy Test {self.proxy_ip}/{self.url}, results was {self.result}, test time={str(self.test_time)}, response_time={self.response_time})'

class Proxy():
    DEFAULT_HEADERS = {'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.5005.134 YaBrowser/22.7.1.828 (beta) Yowser/2.5 Safari/537.36', "Referer": "http://example.com"}
    DEFAULT_TEST_URL = 'https://www.researchgate.net/'

    def __init__(self, ip: str, port: str, proxy_class: str = 'unknown', status:str = 'untested'#, 
                # tests: list() = list()
                ):
        self.ip = ip
        self.proxy_class = proxy_class
        self.port = port
        self.status = status
        self.tests = list() # IDNY but this is the only way it kinda works, except whe it does not
        #print(self.tests)
    
    @property
    def full(self):
        return f'{self.ip}:{self.port}'
    
    @property
    def http(self):
        return f"http://{self.full}"
    
    @property
    def proxy_dict(self):
        proxies = {
                      "http"  : self.http,
                      "https"  : self.http
                  }
        return proxies
    
    def __repr__(self):
        return f'Proxy {self.full}, status is {self.status}'
    def __str__(self):
        return f'Proxy {self.full}, status is {self.status}'
    
    def test(self, test_url=DEFAULT_TEST_URL, headers=DEFAULT_HEADERS, timeout=10, verbose=True):
        check_result = None
        try:
            response = requests.get(test_url, headers=headers, proxies=self.proxy_dict, timeout=timeout)
            check_result = response.status_code
            response_time = response.elapsed.total_seconds()
            if check_result == 200:
                self.status = 'OK'
            else:
                if self.status == 'untested': self.status = 'BAD'
        except Exception as ex:
            check_result = f'ERROR: {ex}'
            response_time = None
            if self.status == 'untested': self.status = 'BAD'
        pr_test = ProxyTest(self.ip, test_url, check_result, test_time=datetime.now(), response_time=response_time)
        self.tests.append(pr_test)
        if verbose: print(f'Tested {self.full} {check_result}')
        if verbose: print(self.tests)
        return self

# gets a list of free proxies from https://free-proxy-list.net/ tests every one in test_url and spits out the fastest
def get_ok_proxy(test_url=DEFAULT_TEST_URL):
    print(f"Getting fastest proxy to access {test_url}")
    ok_proxies, fastest_proxy = get_ok_proxies(test_url)
    return fastest_proxy.proxy_dict        



def get_ok_proxies(test_url=DEFAULT_TEST_URL, verbose=True):
    proxies = get_proxy_list()
    anon_proxies = filter_anon_proxies(proxies)
    print(f"There are {len(anon_proxies)} anon proxies available")
    queue = Queue()
    threads_list = list()
    
    def test_proxy(ip, port, proxy_class, test_url):
        proxy = Proxy(ip, port, proxy_class)
        proxy.test(test_url, verbose=verbose)
        return proxy
    
    for index, row in anon_proxies.iterrows():
        ip, port = get_ip_port_from_row(row)
        thr = threading.Thread(target=lambda q: q.put(test_proxy(ip, port, 'anon', test_url)), args=(queue,))
        thr.start()
        threads_list.append(thr)
    for t in threads_list:
        t.join()
    
    fastest_proxy = None
    fastest_time = 11
    ok_proxies = list()
    while not queue.empty():
        proxy = queue.get()
        if proxy.status == 'OK':
            if verbose: print(proxy)
            if verbose: print(proxy.tests[-1])
            if proxy.tests[-1].response_time < fastest_time: 
                fastest_proxy = proxy
                fastest_time = proxy.tests[-1].response_time
            ok_proxies += proxy,
        last_proxy=proxy
        #if test_proxy(proxy, port, test_url):
        #    print("success")
        #    return formulate_proxy_dict(row)
    if len(ok_proxies) < 1:
        print('NO WORKING PROXIES FOUND, returning non-working')
        return list(), last_proxy
    print(f"Fastest proxy to respond ({fastest_proxy.tests[-1].response_time} sec): {fastest_proxy.full}, total working {len(ok_proxies)} proxies")
    return ok_proxies, fastest_proxy        

def test_proxy(proxy_ip, proxy_port, test_url=DEFAULT_TEST_URL):
    # Добавить Multithreading
    http_proxy = f"http://{proxy_ip}:{proxy_port}"
    proxies = {
                  "http"  : http_proxy,
                  "https"  : http_proxy
              }
    
    try:
        response = requests.get(test_url, headers=headers, proxies=proxies, timeout=10)
        #print(response.text)
        
        if response.status_code == 200:
            http_ok = True
            print(f'tested {proxy_ip}:{proxy_port} OK')
        else:
            http_ok = False
            #print(f'testing {proxy_ip}:{proxy_port} BAD, response "{response.status_code}"')
    except Exception as ex:
        #print(f'testing {proxy_ip}:{proxy_port} BAD, "{ex}"')
        http_ok = False
    return proxy_ip, proxy_port, http_ok

# gets a list of proxies from https://free-proxy-list.net/
def get_proxy_list():
    proxy_page = requests.get('https://free-proxy-list.net/')
    html = lxmlhtml.fromstring(proxy_page.text)
    proxy_all = html.xpath('//*[@id="list"]/div/div[2]/div/table/tbody/tr')
    proxies_df = pd.DataFrame()
    for i, proxy in enumerate(proxy_all):
        columns = ['IP Address', 'Port', 'Code', 'Country', 'Anonymity', 'Google', 'Https', 'Last Checked']
        row_html = proxy.findall("td")
        for cell, column in zip(row_html, columns):
            #print(column, cell.text)
            proxies_df.loc[i, column] = cell.text
    return proxies_df

def filter_anon_proxies(proxies_df):
    anon_proxies = proxies_df[proxies_df['Anonymity'].isin(['elite proxy', 'anonymous'])]
    return anon_proxies


def test_proxy(proxy_ip, proxy_port, test_url=DEFAULT_TEST_URL):
    # Добавить Multithreading
    print(f'testing {proxy_ip}:{proxy_port}')
    http_proxy = f"http://{proxy_ip}:{proxy_port}"
    proxies = {
                  "http"  : http_proxy,
                  "https"  : http_proxy
              }
    
    try:
        response = requests.get(test_url, headers=headers, proxies=proxies, timeout=10)
        #print(response.text)
        if response.status_code == 200:
            http_ok = True
        else:
            http_ok = False
    except Exception:
        http_ok = False
    return http_ok
    
def get_random_proxy(ok_proxies, used=None):
    # НАДО ДОБАВИТЬ ИМЕННО РОТАЦИЮ
    random_proxy_row = ok_proxies.sample(n=1).iloc[0]
    return formulate_proxy_dict(random_proxy_row)

def formulate_proxy_dict(proxy_row):
    proxy_ip, proxy_port = get_ip_port_from_row(proxy_row)
    http_proxy = f"http://{proxy_ip}:{proxy_port}"
    proxies = {
                  "http"  : http_proxy,
                  "https"  : http_proxy
              }
    return proxies

def get_ip_port_from_row(proxy_row):
    proxy_ip = proxy_row['IP Address']
    proxy_port = proxy_row['Port']
    return proxy_ip, proxy_port

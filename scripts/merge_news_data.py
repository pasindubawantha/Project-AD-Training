import json
import pandas
import math
import sys
import os
import numpy as np
import re
import helpers
import detector.detector as detector
import detector.confusion_metrics as confusion_metrics
from datetime import datetime
from datetime import timedelta 
import csv


# args
input_directory = "../data/lseg"
new_file = "../data/lse-news.csv"
label_window_half = 2
data_time_format = '%b %d, %Y'
# data_time_format = '%Y-%m-%d'


news_dataframe = pandas.read_csv(new_file)
news_timestamp = np.array(news_dataframe['timestamp'])

news_timestamp = helpers.string_to_date(news_timestamp, '%Y-%m-%d')

data = {'title':np.array(news_dataframe['title']),
        'source':np.array(news_dataframe['source']),
        'url':np.array(news_dataframe['url']),
        'market_closed':np.zeros(len(news_timestamp))}
news_dataframe = pandas.DataFrame(data, index=news_timestamp)
news_dataframe.index.name = "timestamp"
news_dataframe = news_dataframe[['title',
                        'source',
                        'url',
                        'market_closed']]        


# get all csv files in input directory
reg_x = re.compile(r'\.(csv)')
csv_input_files = []
for path, dnames, fnames in os.walk(input_directory):
    csv_input_files.extend([os.path.join(path, f) for f in fnames if reg_x.search(f)])
csv_input_files.sort()



for input_file in csv_input_files:
    print("Processing " + input_file)
    
    input_dataframe = pandas.read_csv(input_file)

    Close = np.array(input_dataframe['Close'])
    Open = np.array(input_dataframe['Open'])
    High = np.array(input_dataframe['High'])
    Low = np.array(input_dataframe['Low'])
    Vol = np.array(input_dataframe['Vol'])
    timestamp = np.array(input_dataframe['timestamp'])
    label = np.zeros(len(Close))
    timestamp = helpers.string_to_date(timestamp, data_time_format)

    filler_string = []
    for i in range(0,len(label)):
        filler_string.append('')    

    data = {'Close':Close,
            'Open':Open,
            'High':High,
            'Low':Low,
            'Vol':Vol,
            'label':label,
            'label_title':filler_string,
            'label_url':filler_string,
            'label_source':filler_string}
    out_dataframe = pandas.DataFrame(data, index=timestamp)
    out_dataframe.index.name = "timestamp"
    out_dataframe = out_dataframe[['Close',
                            'Open',
                            'High',
                            'Low',
                            'Vol',
                            'label',
                            'label_title',
                            'label_url',
                            'label_source']]
    

    news_timestamp = news_dataframe.index

    for i in news_timestamp:
        title = news_dataframe.at[i, 'title']
        if not isinstance(title, str):
            title = helpers.list_to_string(title, ';')
        else :
            title = helpers.sanitize_str(title)
        url = news_dataframe.at[i, 'url']
        if not isinstance(url, str):
            url = helpers.list_to_string(url, ';')
        else :
            url = helpers.sanitize_str(url)
        source = news_dataframe.at[i, 'source']
        if not isinstance(source, str):
            source = helpers.list_to_string(source, ';')
        else :
            source = helpers.sanitize_str(source)
        
        for j in range(0,label_window_half):
            if i - timedelta(days=j) in out_dataframe.index :
                out_dataframe.at[i - timedelta(days=j), 'label'] = 1.0
                out_dataframe.at[i - timedelta(days=j), 'label_title'] = title
                out_dataframe.at[i - timedelta(days=j), 'label_url'] = url
                out_dataframe.at[i - timedelta(days=j), 'label_source'] = source
                # print("label at " + str(i - timedelta(days=j)))
            if i + timedelta(days=j) in out_dataframe.index :
                out_dataframe.at[i + timedelta(days=j), 'label'] = 1.0
                out_dataframe.at[i + timedelta(days=j), 'label_title'] = title
                out_dataframe.at[i + timedelta(days=j), 'label_url'] = url
                out_dataframe.at[i + timedelta(days=j), 'label_source'] = source
                # print("label at " + str(i + timedelta(days=j)))
        if not (i in out_dataframe.index) :
            news_dataframe.at[i, 'market_closed'] = 1.0
    
    # out_dataframe.update(out_dataframe[['label_title', 'label_url', 'label_source']].applymap('"{}"'.format))
    out_dataframe.to_csv(input_file)


news_dataframe.to_csv(new_file)
print("All Done !")




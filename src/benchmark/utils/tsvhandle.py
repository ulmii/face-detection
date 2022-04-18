#! /usr/bin/env python3

import csv
from datetime import datetime
from .envmetadata import EnvMetadata
from .envload import EnvLoad
from pathlib import Path

class TsvHandle(object):
    def __init__(self, name):
        self.name = name

        Path("./results").mkdir(parents=True, exist_ok=True)

        self.env = EnvMetadata()
        self.load = EnvLoad()

        time = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        
        raw_file = "./results/{}-{}".format(name, time)
        self.file_path = "{}.tsv".format(raw_file)
        self.ap_file_path = "{}-ap.tsv".format(raw_file)
        self.metadata_path = "{}-metadata.tsv".format(raw_file)
        self.env_load_path = "{}-load.tsv".format(raw_file)
        
        with open(self.file_path, 'w', newline='\n') as tsvfile:
            writer = csv.writer(tsvfile, delimiter=str('\t'))
            writer.writerow(self.get_headers())

        with open(self.ap_file_path, 'w', newline='\n') as tsvfile:
            writer = csv.writer(tsvfile, delimiter=str('\t'))
            writer.writerow(self.get_ap_headers())
        
        with open(self.metadata_path, 'w', newline='\n') as tsvfile:
            writer = csv.writer(tsvfile, delimiter=str('\t'))
            writer.writerow(self.env.get_headers())
            writer.writerow(self.env.get_data())

        with open(self.env_load_path, 'w', newline='\n') as tsvfile:
            writer = csv.writer(tsvfile, delimiter=str('\t'))
            writer.writerow(self.load.get_headers())
            writer.writerow(self.load.get_data())

        self.__last_update = datetime.utcnow()

    def __enter__(self):
        self.file = open(self.file_path, 'a', newline='\n')
        self.writer = csv.writer(self.file, delimiter=str('\t'))

        self.ap_file = open(self.ap_file_path, 'a', newline='\n')
        self.ap_writer = csv.writer(self.ap_file, delimiter=str('\t'))        

        self.load_file = open(self.env_load_path, 'a', newline='\n')
        self.load_writer = csv.writer(self.load_file, delimiter=str('\t'))

        return self
    
    def get_headers(self):
        return ['Timestamp', 'Speed', 'Precision', 'Recall', 'F1_Score', 'Ious', 'Positives', 'False_Positives', 'Negatives', 'Num_Of_Faces','Predicted', 'Ground_Truth']
    
    def get_ap_headers(self):
        return ['Confidence', 'TP_FP@25', 'TP_FP@50', 'TP_FP@75']
    
    def get_file_path(self):
        return self.file_path
    
    def get_metadata_path(self):
        return self.metadata_path

    def append_ap(self, row):
        self.ap_writer.writerow(row)

    def append(self, row):
        self.writer.writerow(row)

        interval = datetime.utcnow() - self.__last_update

        if interval.seconds > 10:
            self.append_load()

    def append_load(self, interval = None):
        self.load.update_load(interval)
        self.load_writer.writerow(self.load.get_data())
        self.__last_update = datetime.utcnow()

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            print(exc_type, exc_value, tb)
        
        self.writer = None
        self.ap_writer = None
        self.load_writer = None
        self.file.close()
        self.ap_file.close()
        self.load_file.close()
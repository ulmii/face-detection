#! /usr/bin/env python3

import csv
from datetime import datetime
from .envmetadata import EnvMetadata
from pathlib import Path

class TsvHandle:
    def __init__(self, name):
        self.name = name

        Path("./results").mkdir(parents=True, exist_ok=True)

        self.env = EnvMetadata()
        
        time = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        
        raw_file = "./results/{}-{}".format(name, time)
        file_path = "{}.tsv".format(raw_file)
        metadata_path = "{}-metadata.tsv".format(raw_file)
        
        with open(file_path, 'w', newline='\n') as tsvfile:
            writer = csv.writer(tsvfile, delimiter=str('\t'))
            writer.writerow(self.get_headers())
        
        with open(metadata_path, 'w', newline='\n') as tsvfile:
            writer = csv.writer(tsvfile, delimiter=str('\t'))
            writer.writerow(self.env.get_headers())
        
        self.file_path = file_path
        self.metadata_path = metadata_path
    
    def get_headers(self):
        return ['Timestamp', 'Speed', 'Precision', 'Recall', 'Ious', 'Positives', 'False_Positives', 'Negatives', 'Predicted', 'Ground_Truth']
    
    def get_file_path(self):
        return self.file_path
    
    def get_metadata_path(self):
        return self.metadata_path

    # TODO: Append to metadata and data
    def append(row_dict):
        print(row_dict.keys())
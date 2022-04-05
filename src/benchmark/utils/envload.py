#! /usr/bin/env python3

import psutil
from datetime import datetime 

class EnvLoad:
    def __init__(self):
        self.cpu_load = psutil.cpu_percent(interval=10)
        self.avail_mem = psutil.virtual_memory().available
    
    def get_headers(self):
        return ["Timestamp", "CPU_Load", "RAM_Avail"]

    def get_data(self):
        return [datetime.utcnow().isoformat(), self.cpu_load, self.avail_mem]

    def update_load(self, interval = None):
        self.cpu_load = self.cpu_load = psutil.cpu_percent(interval=interval)
        self.avail_mem = psutil.virtual_memory().available
        
    def __str__(self):
        return "({}, {:.2f}GB)".format(self.cpu_load, self.avail_mem * 1e-9)
    
    def __repr__(self):
        return "EnvLoad({}, {:.2f}GB)".format(self.cpu_load, self.avail_mem * 1e-9)
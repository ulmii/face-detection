#! /usr/bin/env python3

import cpuinfo

from tensorflow.python.client import device_lib
import tensorflow as tf

class EnvMetadata:
    def __init__(self):
        self.python = cpuinfo.get_cpu_info()['python_version']
        self.cpu_model = cpuinfo.get_cpu_info()['brand_raw']
        self.gpus = self.get_available_gpus()
        
    def get_available_gpus(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.compat.v1.Session(config=config)
        local_device_protos = device_lib.list_local_devices()
        
        return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']
    
    def get_headers(self):
        return ["Python", "CPU", "GPUs"]

    def get_data(self):
        return [self.python, self.cpu_model, self.gpus]

    def __str__(self):
        return "({}, {}, {})".format(self.python, self.cpu_model, self.gpus)
    
    def __repr__(self):
        return "EnvMetadata({}, {}, {})".format(self.python, self.cpu_model, self.gpus)

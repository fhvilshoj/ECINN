import os
import configparser

def cf_dir():     
    if os.path.isfile('config.ini'): 
        cfg = configparser.ConfigParser()
        cfg.read('config.ini')
        return cfg['data']['cf_output_dir']
    else: 
        return './counterfactual_examples'

class IncrementIndex:
    def __init__(self):
        self._cnts = {}

    def next(self, key):
        if key not in self._cnts:
            self._cnts[key] = 0
        n = self._cnts[key]
        self._cnts[key] += 1
        return n


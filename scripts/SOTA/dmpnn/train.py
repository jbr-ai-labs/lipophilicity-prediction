"""Trains a chemprop model on a dataset."""

import sys


import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
chemprop_dir = os.path.join(currentdir, 'chemprop')

sys.path.insert(0,chemprop_dir) 

from train import chemprop_train
import pandas as pd
import os
import numpy as np
import yaml

if __name__ == '__main__':
    chemprop_train()

from __future__ import print_function

import os
import pickle
import csv

import numpy as np

from topology import Topology

import aggregations
from cifar10.cifar10_normal_train import *

from cifar10.cifar10_models import *

from cifar10.sgd import SGD

from arguments import Arguments

from attack import min_max_attack
from client import Client

def federated_learning(args):
    
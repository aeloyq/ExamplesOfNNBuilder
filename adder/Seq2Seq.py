# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:44:11 2016

@author: aeloyq
"""
from nnbuilder import *
from nnbuilder.tools import loaddatas

### Set Global Config ###
config.set(name='imdb', max_epoch=5000, batch_size=16, valid_batch_size=64)

### Set Extension Config ###
monitor.config.set(report_iter_frequence=25)
saveload.config.set(load=False)

### Set Optimizer Config ###
gradientdescent.adam.set(learning_rate=0.0001)

### Build Model ###
Lstm = model(10000, var.X.sequence, var.Y.catglory)
Lstm.sequential()
Lstm.add(embedding(128))
Lstm.add(dropout(0.8))
Lstm.add(lstm(128, out='mean'))
Lstm.add(normalization(method='layer'))
Lstm.add(dropout(0.8))
Lstm.add(softmax(2))
Lstm.add(dropout(0.8))

### Load Data ###
data = loaddatas.Load_imdb('./datasets/imdb.pkl', maxlen=100)

### Fit Model ###
train(data=data, model=Lstm, optimizer=gradientdescent.adam, extensions=[monitor, earlystop])

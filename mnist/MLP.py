# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:44:11 2016

@author: aeloyq
"""

from nnbuilder import *

### Set Global Config ###
config.set(name='mnist', max_epoch=5, batch_size=20, valid_batch_size=20)

### Set Extension Config ###
monitor.config.set(report_iter=True, report_iter_frequence=500)
earlystop.config.set(valid_epoch=True)
sample.config.set(sample_from = 'valid',sample_func = tools.samples.mnist_sample)
saveload.config.set(load = False)

### Build Model ###
Mlp = model(28 * 28)
Mlp.add(hnn(500), 'hidden')
Mlp.add(softmax(10), 'output')
Mlp.add(regularization())

### Load Data ###
data = tools.loaddatas.Load_mnist("./datasets/mnist.pkl.gz")

### Fit Model ###
train(data=data, model=Mlp, optimizer=gradientdescent.sgd,
      extensions=[monitor, earlystop, sample, shuffledata, saveload])
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:44:11 2016

@author: aeloyq
"""

from nnbuilder import *


config.name= 'mnist'
config.data_path= "../datasets/mnist.pkl.gz"
config.max_epoches=5
config.valid_batch_size=20
config.batch_size=20

sample.config.sample_func=samples.mnist_sample
monitor.config.plot=True
monitor.config.report_iter=True
monitor.config.report_iter_frequence=500
earlystop.config.valid_epoch=True
sample.config.sample_from='valid'
saveload.config.save_epoch=True

datastream  = Load_mnist()

model = model(28*28)
model.add(hiddenlayer(500),'hidden')
model.add(softmax(10),'output')
model.add(regularization())


train(datastream=datastream,model=model,algrithm=sgd, extension=[monitor,earlystop,sample,saveload])
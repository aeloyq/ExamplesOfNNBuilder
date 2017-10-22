# -*- coding: utf-8 -*-
"""
Created on  十月 16 18:46 2017

@author: aeloyq
"""

from nnbuilder import *

### Set Global Config ###
config.set(name='mnist-convnet-sgd', max_epoch=500, batch_size=1000, valid_batch_size=1000)

### Set Optimizer Config ###
gradientdescent.nadam.config.set(learning_rate=0.1)

### Set Extension Config ###
monitor.config.set(report_epoch_frequence=50)
earlystop.config.set(valid_freq=1000, valid_epoch=False)
sample.config.set(sample_from='valid', sample_func=tools.samples.image_sample)
saveload.config.set(load=False)

### Build Model ###
ConvNet = model((1, 28, 28), X=var.X.image)
ConvNet.add(conv(nfilters=6, filtersize=[5, 5]))
ConvNet.add(subsample(windowsize=(2, 2)))
ConvNet.add(asymconv(filters=[(3, 3), (4, 2), (2, 1), (6, 0)], filtersize=[5, 5]))
ConvNet.add(subsample(windowsize=(2, 2)))
ConvNet.add(conv(nfilters=120, filtersize=[4, 4]))
ConvNet.add(flatten())
ConvNet.add(hnn(84))
ConvNet.add(softmax(10))

### Load Data ###
data = tools.loaddatas.Load_mnist_image("./datasets/mnist.pkl.gz")

### Fit Model ###
train(data=data, model=ConvNet, optimizer=gradientdescent.nadam,
      extensions=[monitor, earlystop, saveload])

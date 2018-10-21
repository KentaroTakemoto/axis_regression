import math

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import serializers
from chainer import Variable
from chainer.initializers import HeNormal

class VGG_double(chainer.Chain):
    def __init__(self, n_out=7):
        super(VGG_double, self).__init__(
            conv = L.Convolution2D(6,3,ksize=3,stride=1,pad=1,initialW=HeNormal()),
            model = L.VGG16Layers(),
            fc = L.Linear(4096,n_out))

    def __call__(self, x, t=None, train=True, finetune=False):
        with chainer.using_config('enable_backprop', finetune):
            x = Variable(self.xp.asarray(x.data,dtype=np.float32))
        x = F.relu(self.conv(x))
        with chainer.using_config('train', finetune):
            h = self.model(x,layers=['fc7'])['fc7']
        if finetune==False:
            with chainer.using_config('enable_backprop', train):
                h = Variable(h.data)
        h = F.sigmoid(self.fc(h))
        if train:
            return F.mean_squared_error(h, t)
        else:
            return h

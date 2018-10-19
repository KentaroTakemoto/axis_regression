import math

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import serializers
from chainer import Variable

class VGG_double(chainer.Chain):
    def __init__(self, n_out=7):
        super(VGG_double, self).__init__(
            model = L.VGG16Layers(),
            fc = L.Linear(8192,n_out))
        )
        self.train = False

    def __call__(self, x, t=None, train=True, finetune=False):
        x1 = Variable(self.xp.asarray(x[:,:3,:,:],dtype=np.float32),volatile=not finetune)
        x2 = Variable(self.xp.asarray(x[:,3:,:,:],dtype=np.float32),volatile=not finetune)
        h1 = self.model(x1,layers=['fc7'], test=not finetune)['fc7']
        h2 = self.model(x2,layers=['fc7'], test=not finetune)['fc7']
        h = F.concat([h1, h2], axis=1)
        if finetune==False:
            h = Variable(h.data,volatile=not train)
        h = F.sigmoid(self.fc(h))
        if self.train:
            return F.mean_squared_error(h, t)
        else:
            return h

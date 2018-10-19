import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, Variable
from chainer import training
from chainer.training import extensions

import sys
import os
import argparse

from model import VGG_double
from preprocess import load_data

parser = argparse.ArgumentParser(description='Chainer Axis Regression Network')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--test_angles', '-te', default='60,165', type=str)
parser.add_argument('--train_view_params', '-tr', default=90, type=int)
parser.add_argument('--view_params_file', '-vi', default='~/other_githubs/RenderForCNN/test_results/view_params.txt', type=str)
parser.add_argument('--batchsize', '-b', type=int, default=20,
                    help='batch size (default value is 20)')
parser.add_argument('--initmodel', '-i', default=None, type=str,
                    help='initialize the model from given file')
parser.add_argument('--epoch', '-e', default=100, type=int)
parser.add_argument('--lr', '-l', default=1e-5, type=float)
parser.add_argument('--image_size', default=320, type=int)
parser.add_argument('--output', default=7, type=int)
args = parser.parse_args()

n_epoch = args.epoch
n_out = args.output
batchsize = args.batchsize
image_size = args.image_size
initmodel = args.initmodel
test_angles = args.test_angles.split(',')
train_view_params = args.train_view_params
view_params_file = args.view_params_file

with open(view_params_file,"r") as f:
    ls = f.readlines()
view_params = [l.rstrip('\n') for l in ls]
n_iter = len(view_params) * 50 // batchsize
gpu_flag = True if args.gpu >= 0 else False

model = VGG_double(n_out)

if initmodel is not None:
    serializers.load_npz(initmodel, model)
    print("Load initial weight from {}".format(initmodel))

if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()
    model.to_gpu()

xp = np if args.gpu < 0 else cuda.cupy

# Setup optimizer parameters.
optimizer = optimizers.Adam(alpha=args.lr)
optimizer.setup(model)
# optimizer.add_hook(chainer.optimizer.WeightDecay(1e-7), 'hook_vgg')

print("## INFORMATION ##")
print("Num Epoch: {}, Data: {}, Batchsize: {}, Iteration {}".format(n_epoch, len(view_params) * 50, batchsize, n_iter))

print("-"*40)
for epoch in range(n_epoch):
    print('epoch', epoch+1)
    for i in range(n_iter):
        model.zerograds()
        indices = range(i * batchsize, (i+1) * batchsize)

        x = xp.zeros((batchsize, 6, image_size, image_size), dtype=xp.float32)
        y = xp.zeros((batchsize, n_out), dtype=xp.float32)
        for j in range(batchsize):
            view_param = view_params[np.random.randint(train_view_params)]
            model_number = np.random.randint(2)
            x[j] = load_data(view_param, model_number, mode="data", test_angles=test_angles, image_size=image_size, xp=xp)
            y[j] = load_data(view_param, model_number, mode="label",n_out=n_out, xp=xp)
        x = Variable(x)
        y = Variable(y)
        if epoch < n_epoch/2:
            loss = model(x, y, train=True, finetune=False)
        else:
            loss = model(x, y, train=True, finetune=True)

        sys.stdout.write("\r%s" % "batch: {}/{}, loss: {}".format(i+1, n_iter, loss.data))
        sys.stdout.flush()

        loss.backward()
        optimizer.update()
    print("\n"+"-"*40)

if not os.path.exists("weight"):
    os.mkdir("weight")
serializers.save_npz('weight/test1.weight', model)
serializers.save_npz('weight/test1.state', optimizer)
print('save weight')

from chainer import serializers
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, Variable
from chainer import training
from chainer.training import extensions
import os
import argparse

from model import VGG_double
from preprocess import load_data, select_angles


parser = argparse.ArgumentParser(description='Chainer Axis Regression Network')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--test_angles', '-te', default='60,165', type=str)
parser.add_argument('--train_view_params', '-tr', default=90, type=int)
parser.add_argument('--view_params_file', '-vi', default='/home/mil/takemoto/other_githubs/RenderForCNN/test_results/view_params.txt', type=str)
parser.add_argument('--weight', '-w', default="weight/test1.weight", type=str)
parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--output', default=7, type=int)
args = parser.parse_args()

print(args.weight)
train_view_params = args.train_view_params
view_params_file = args.view_params_file

with open(view_params_file,"r") as f:
    ls = f.readlines()
view_params = [l.rstrip('\n') for l in ls]

model = VGG_double(n_out=args.output)
serializers.load_npz(args.weight, model)

if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

preds = []
labels = []
for i in range(100):
    test_angles = args.test_angles.split(',')
    view_param = view_params[np.random.randint(len(view_params)-train_view_params) + train_view_params]
    model_number = np.random.randint(2)
    if model_number==1:
        test_angles = [str(int(test_angles[0])*-1), str(int(test_angles[1])*-1)]
    x = load_data(view_param, model_number, mode="data", angles=test_angles, size=args.image_size, xp=xp)
    y = load_data(view_param, model_number, mode="label",n_out=args.output, xp=np)
    x = xp.expand_dims(x, axis=0)
    x = Variable(x)
    pred = chainer.cuda.to_cpu(model(x, train=False).data[0])
    preds.append(pred)
    labels.append(chainer.cuda.to_cpu(y))

print(preds[0])
print(labels[0])
preds = chainer.cuda.to_cpu(np.array(preds)).astype(np.float32)

labels = chainer.cuda.to_cpu(np.array(labels)).astype(np.float32)

# if args.output == 7:
#     print('l2 error for direction : {}'.format(np.mean(np.linalg.norm(preds[:,:3]-labels[:,:3],axis=1))))
#     print('l2 error for location : {}'.format(np.mean(np.linalg.norm(preds[:,3:]-labels[:,3:],axis=1))))
# print('whole l2 error : {}'.format(np.mean(np.linalg.norm(preds-labels,axis=1))))

if args.output == 7:
    print('l2 error for direction : {}'.format(np.mean((preds[:,:3]-labels[:,:3])**2)))
    print('l2 error for location : {}'.format(np.mean((preds[:,3:]-labels[:,3:])**2)))
print('whole l2 error : {}'.format(np.mean((preds-labels)**2,)))

# print(F.mean_squared_error(preds[:,:3],labels[:,:3]))
# if args.output == 7:
#     print('l2 error for direction : {}'.format(F.mean_squared_error(preds[:,:3],labels[:,:3])))
#     print('l2 error for location : {}'.format(F.mean_squared_error(preds[:,3:],labels[:,3:])))
# print('whole l2 error : {}'.format(F.mean_squared_error(preds,labels)))

print('-'*80)

from chainer import serializers
import numpy as np
from PIL import Image
import os
import argparse
import cv2

from model import VGG_double
from preprocess import load_data, select_angles


parser = argparse.ArgumentParser(description='Chainer Axis Regression Network')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--test_angles', '-te', default='60,165', type=str)
parser.add_argument('--train_view_params', '-tr', default=90, type=int)
parser.add_argument('--weight', '-w', default="weight/test1.weight", type=str)
parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--output', default=7, type=int)
args = parser.parse_args()

train_view_params = args.train_view_params
test_angles = args.test_angles.split(',')

model = VGG_double(n_out=args.output)
serializers.load_npz(args.weight, model)

preds = []
labels = []
for i in range(100):
    view_param = view_params[np.random.randint(len(view_params)-train_view_params) + train_view_params]
    model_number = np.random.randint(2)
    x = load_data(view_param, model_number, mode="data", angles=test_angles, size=image_size, xp=xp)
    y = load_data(view_param, model_number, mode="label",n_out=n_out, xp=xp)
    x = np.expand_dims(x, axis=0)
    pred = model(x).data[0]
    preds.append(pred)
    labels.append(y)

print(preds)
print('-'*40)
print(labels)

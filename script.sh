#!/bin/sh

# python train.py --gpu 1 -b 20 -e 100 --output 7 -s test1
# python train.py --gpu 1 -b 20 -e 100 --output 3 -s test2
# python train.py --gpu 1 -b 20 -e 100 --output 4 -s test3

python train_softmax.py --gpu 1 -b 20 -e 100 --output 3 -s test4

# python test.py --gpu 1 -w weight/test1.weight --output 7
# python test.py --gpu 1 -w weight/test2.weight --output 3
# python test.py --gpu 1 -w weight/test3.weight --output 4

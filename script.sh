#!/bin/sh

# python train.py --gpu 1 -b 20 -e 100 --output 7 -s test1
# python train.py --gpu 1 -b 20 -e 100 --output 3 -s test2
# python train.py --gpu 1 -b 20 -e 100 --output 4 -s test3

python test.py --gpu 2 -w weight/test1.weight --output 7

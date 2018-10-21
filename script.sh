#!/bin/sh

# python train.py --gpu 1 -b 20 -e 100 --output 7 -s test1
# python train.py --gpu 1 -b 20 -e 100 --output 3 -s test2
# python train.py --gpu 1 -b 20 -e 100 --output 4 -s test3

# python train_softmax.py --gpu 1 -b 20 -e 100 --output 3 -s test4
python train_conv.py --gpu 3 -b 20 -e 100 --output 7 -s test5


# python test.py --gpu 1 -w weight/test1.weight --output 7
# python test.py --gpu 1 -w weight/test2.weight --output 3
# python test.py --gpu 1 -w weight/test3.weight --output 4
# python test_softmax.py --gpu 1 -w weight/test4.weight --output 3
# python test_conv.py --gpu 1 -w weight/test5.weight --output 7

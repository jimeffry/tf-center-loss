#!/usr/bin/env bash
python train.py  --batch_size 32  --epoch_num 10000 --lr 0.01  --train_file ../data/Ms_1M_train.txt --val_file  ../data/Ms_1M_val.txt
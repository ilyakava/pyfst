#!/bin/bash

rm -rf /scratch0/ilya/locDoc/pyfst/models/throw;
# CUDA_VISIBLE_DEVICES=1 \
# python hyper_pixelNN.py \
# --dataset IP \
# --npca_components 4 \
# --attribute_profile \
# --batch_size 50 \
# --lr 0.00005 \
# --network aptoula \
# --network_spatial_size 9 \
# --eval_period 25 \
# --num_epochs 100000 \
# --mask_root  /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/IP_distributed_trainval_s05_4_387762.mat \
# --model_root /scratch0/ilya/locDoc/pyfst/models/eap/IP_distributed_trainval_s05_4_387762/0/ \
# --predict

CUDA_VISIBLE_DEVICES=1 \
python hyper_pixelNN.py \
--dataset IP \
--npca_components 4 \
--attribute_profile \
--batch_size 50 \
--lr 0.00001 \
--network aptoula \
--network_spatial_size 9 \
--eval_period 25 \
--num_epochs 100000 \
--mask_root  /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/IP_strictsinglesite_trainval_p0200_0_537986.mat \
--model_root /scratch0/ilya/locDoc/pyfst/models/throw
# --predict
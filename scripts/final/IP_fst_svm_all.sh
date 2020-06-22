#!/bin/bash

CUDA_VISIBLE_DEVICES=1 \
python hyper_pixelNN.py \
--dataset IP \
--svm_multi_mask_file_list ./mask_lists/IP_sss.txt \
--model_root /scratch1/ilya/locDoc/pyfst/june_models/ip_sss_fst \
--preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/ip_sss_fst/preprocessed.npz \
--st_type IP_SSS \
--fst_preprocessing \
--network_spatial_size 1 \
--batch_size 1000 \
--svm_regularization_param 1000

CUDA_VISIBLE_DEVICES=1 \
python hyper_pixelNN.py \
--dataset IP \
--svm_multi_mask_file_list ./mask_lists/IP_distributed.txt \
--model_root /scratch1/ilya/locDoc/pyfst/june_models/ip_dist_fst \
--preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/ip_dist_fst/preprocessed.npz \
--st_type IP_dist \
--fst_preprocessing \
--network_spatial_size 1 \
--batch_size 1000 \
--svm_regularization_param 1000

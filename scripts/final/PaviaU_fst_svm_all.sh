#!/bin/bash

# SSS
CUDA_VISIBLE_DEVICES=1 \
python hyper_pixelNN.py \
--dataset PaviaU \
--svm_multi_mask_file_list ./mask_lists/PaviaU_sss.txt \
--model_root /scratch1/ilya/locDoc/pyfst/june_models/paviau_sss_fst \
--preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/paviau_sss_fst/preprocessed.npz \
--st_type paviaU_SSS \
--fst_preprocessing \
--network_spatial_size 1 \
--batch_size 1000 \
--svm_regularization_param 1000


CUDA_VISIBLE_DEVICES=1 \
python hyper_pixelNN.py \
--dataset PaviaU \
--svm_multi_mask_file_list ./mask_lists/PaviaU_distributed.txt \
--model_root /scratch1/ilya/locDoc/pyfst/june_models/paviau_dist_fst \
--preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/paviau_dist_fst/preprocessed.npz \
--st_type paviaU_dist \
--fst_preprocessing \
--network_spatial_size 1 \
--batch_size 1000 \
--svm_regularization_param 1000
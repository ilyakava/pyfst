#!/bin/bash

# SSS
CUDA_VISIBLE_DEVICES=1 \
python hyper_pixelNN.py \
--dataset KSC \
--svm_multi_mask_file_list ./mask_lists/KSC_sss.txt \
--model_root /scratch1/ilya/locDoc/pyfst/june_models/ksc_sss_fst \
--preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/ksc_sss_fst/preprocessed.npz \
--st_type KSC_SSS \
--fst_preprocessing \
--network_spatial_size 1 \
--batch_size 1000

CUDA_VISIBLE_DEVICES=1 \
python hyper_pixelNN.py \
--dataset KSC \
--svm_multi_mask_file_list ./mask_lists/KSC_distributed.txt \
--model_root /scratch1/ilya/locDoc/pyfst/june_models/ksc_dist_fst \
--preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/ksc_dist_fst/preprocessed.npz \
--st_type KSC_dist \
--fst_preprocessing \
--network_spatial_size 1 \
--batch_size 1000

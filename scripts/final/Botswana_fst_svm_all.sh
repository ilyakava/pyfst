#!/bin/bash

CUDA_VISIBLE_DEVICES=1 \
python hyper_pixelNN.py \
--dataset Botswana \
--svm_multi_mask_file_list ./mask_lists/Botswana_sss.txt \
--model_root /scratch1/ilya/locDoc/pyfst/june_models/bots_sss_fst \
--preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/bots_sss_fst/preprocessed.npz \
--st_type Botswana_SSS \
--fst_preprocessing \
--network_spatial_size 1 \
--batch_size 1000


CUDA_VISIBLE_DEVICES=1 \
python hyper_pixelNN.py \
--dataset Botswana \
--svm_multi_mask_file_list ./mask_lists/Botswana_distributed.txt \
--model_root /scratch1/ilya/locDoc/pyfst/june_models/bots_dist_fst \
--preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/bots_dist_fst/preprocessed.npz \
--st_type Botswana_dist \
--fst_preprocessing \
--network_spatial_size 1 \
--batch_size 1000
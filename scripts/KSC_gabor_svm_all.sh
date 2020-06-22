#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
python hyper_pixelNN.py \
--dataset KSC \
--svm_multi_mask_file_list ./mask_lists/KSC_sss.1.txt \
--model_root /scratch0/ilya/locDoc/pyfst/june_models/ksc_sss_gabor_svm \
--preprocessed_data_path /scratch0/ilya/locDoc/pyfst/june_models/ksc_sss_gabor_svm/preprocessed.npz \
--st_type KSC2_gabor \
--fst_preprocessing \
--network_spatial_size 1 \
--batch_size 1000
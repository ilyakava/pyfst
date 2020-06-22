#!/bin/bash

# KSC SSS
# CUDA_VISIBLE_DEVICES=0 \
# python hyper_pixelNN.py \
# --dataset KSC \
# --svm_multi_mask_file_list ./mask_lists/KSC_sss.txt \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/ksc_sss_gabor \
# --preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/ksc_sss_gabor/preprocessed.npz \
# --st_type KSC_SSS_gabor \
# --fst_preprocessing \
# --network_spatial_size 1 \
# --batch_size 1000

# # Bots SSS
# CUDA_VISIBLE_DEVICES=0 \
# python hyper_pixelNN.py \
# --dataset Botswana \
# --svm_multi_mask_file_list ./mask_lists/Botswana_sss.txt \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/bots_sss_gabor \
# --preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/bots_sss_gabor/preprocessed.npz \
# --st_type Botswana_SSS_gabor \
# --fst_preprocessing \
# --network_spatial_size 1 \
# --batch_size 1000

# # PaviaU SSS
# CUDA_VISIBLE_DEVICES=0 \
# python hyper_pixelNN.py \
# --dataset PaviaU \
# --svm_multi_mask_file_list ./mask_lists/PaviaU_sss.txt \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/paviau_sss_gabor \
# --preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/paviau_sss_gabor/preprocessed.npz \
# --st_type paviaU_SSS_gabor \
# --fst_preprocessing \
# --network_spatial_size 1 \
# --batch_size 1000

# IP SSS
CUDA_VISIBLE_DEVICES=0 \
python hyper_pixelNN.py \
--dataset IP \
--svm_multi_mask_file_list ./mask_lists/IP_sss.txt \
--model_root /scratch1/ilya/locDoc/pyfst/june_models/ip_sss_gabor \
--preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/ip_sss_gabor/preprocessed.npz \
--st_type IP_SSS_gabor \
--fst_preprocessing \
--network_spatial_size 1 \
--batch_size 1000


# KSC dist
# CUDA_VISIBLE_DEVICES=0 \
# python hyper_pixelNN.py \
# --dataset KSC \
# --svm_multi_mask_file_list ./mask_lists/KSC_distributed.txt \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/ksc_dist_gabor \
# --preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/ksc_dist_gabor/preprocessed.npz \
# --st_type KSC_dist_gabor \
# --fst_preprocessing \
# --network_spatial_size 1 \
# --batch_size 1000

# # Bots dist
# CUDA_VISIBLE_DEVICES=0 \
# python hyper_pixelNN.py \
# --dataset Botswana \
# --svm_multi_mask_file_list ./mask_lists/Botswana_distributed.txt \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/bots_dist_gabor \
# --preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/bots_dist_gabor/preprocessed.npz \
# --st_type Botswana_dist_gabor \
# --fst_preprocessing \
# --network_spatial_size 1 \
# --batch_size 1000

# # PaviaU dist
# CUDA_VISIBLE_DEVICES=0 \
# python hyper_pixelNN.py \
# --dataset PaviaU \
# --svm_multi_mask_file_list ./mask_lists/PaviaU_distributed.txt \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/paviau_dist_gabor \
# --preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/paviau_dist_gabor/preprocessed.npz \
# --st_type paviaU_dist_gabor \
# --fst_preprocessing \
# --network_spatial_size 1 \
# --batch_size 1000

# # IP dist
# CUDA_VISIBLE_DEVICES=0 \
# python hyper_pixelNN.py \
# --dataset IP \
# --svm_multi_mask_file_list ./mask_lists/IP_distributed.txt \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/ip_dist_gabor \
# --preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/ip_dist_gabor/preprocessed.npz \
# --st_type IP_dist_gabor \
# --fst_preprocessing \
# --network_spatial_size 1 \
# --batch_size 1000
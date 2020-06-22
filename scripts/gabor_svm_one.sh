
# ls /scratch1/ilya/locDoc/pyfst/june_models | grep gabor

# bots_dist_gabor
# bots_sss_gabor
# ip_dist_gabor
# ip_sss_gabor
# ksc_dist_gabor
# ksc_sss_gabor
# paviau_dist_gabor
# paviau_sss_gabor

# CUDA_VISIBLE_DEVICES=0 \
# python hyper_pixelNN.py \
# --dataset IP \
# --svm_predict \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/IP_distributed_trainval_s05_7_529916.mat \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/ip_dist_gabor/ \
# --preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/ip_dist_gabor/preprocessed.npz \
# --st_type IP_dist_gabor \
# --fst_preprocessing \
# --network_spatial_size 1 \
# --batch_size 1000

# CUDA_VISIBLE_DEVICES=0 \
# python hyper_pixelNN.py \
# --dataset IP \
# --svm_predict \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/IP_strictsinglesite_trainval_s05_8_351788.mat \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/ip_sss_gabor/ \
# --preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/ip_sss_gabor/preprocessed.npz \
# --st_type IP_SSS_gabor \
# --fst_preprocessing \
# --network_spatial_size 1 \
# --batch_size 1000

# CUDA_VISIBLE_DEVICES=1 \
# python hyper_pixelNN.py \
# --dataset KSC \
# --svm_predict \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/KSC_strictsinglesite_trainval_s50_3_643924.mat \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/ksc_sss_gabor/ \
# --preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/ksc_sss_gabor/preprocessed.npz \
# --st_type KSC_SSS_gabor \
# --fst_preprocessing \
# --network_spatial_size 1 \
# --batch_size 1000

# CUDA_VISIBLE_DEVICES=1 \
# python hyper_pixelNN.py \
# --dataset KSC \
# --svm_predict \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/KSC_distributed_trainval_s50_4_112928.mat \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/ksc_dist_gabor/ \
# --preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/ksc_dist_gabor/preprocessed.npz \
# --st_type KSC_dist_gabor \
# --fst_preprocessing \
# --network_spatial_size 1 \
# --batch_size 1000

# CUDA_VISIBLE_DEVICES=0 \
# python hyper_pixelNN.py \
# --dataset Botswana \
# --svm_predict \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/Botswana_strictsinglesite_trainval_s20_6_929435.mat \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/bots_sss_gabor/ \
# --preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/bots_sss_gabor/preprocessed.npz \
# --st_type Botswana_SSS_gabor \
# --fst_preprocessing \
# --network_spatial_size 1 \
# --batch_size 1000

# CUDA_VISIBLE_DEVICES=1 \
# python hyper_pixelNN.py \
# --dataset Botswana \
# --svm_predict \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/Botswana_distributed_trainval_s20_5_817361.mat \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/bots_dist_gabor/ \
# --preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/bots_dist_gabor/preprocessed.npz \
# --st_type Botswana_dist_gabor \
# --fst_preprocessing \
# --network_spatial_size 1 \
# --batch_size 1000

# CUDA_VISIBLE_DEVICES=0 \
# python hyper_pixelNN.py \
# --dataset PaviaU \
# --svm_predict \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/PaviaU_strictsinglesite_trainval_s90_5_672053.mat \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/paviau_sss_gabor/ \
# --preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/paviau_sss_gabor/preprocessed.npz \
# --st_type paviaU_SSS_gabor \
# --fst_preprocessing \
# --network_spatial_size 1 \
# --batch_size 1000

# CUDA_VISIBLE_DEVICES=0 \
# python hyper_pixelNN.py \
# --dataset PaviaU \
# --svm_predict \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/PaviaU_distributed_trainval_p0200_2_123909.mat \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/paviau_dist_gabor/ \
# --preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/paviau_dist_gabor/preprocessed.npz \
# --st_type paviaU_dist_gabor \
# --fst_preprocessing \
# --network_spatial_size 1 \
# --batch_size 1000

### conf mat

CUDA_VISIBLE_DEVICES=0 \
python hyper_pixelNN.py \
--dataset Botswana \
--svm_predict \
--mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/Botswana_distributed_trainval_s10_8_992993.mat \
--model_root /scratch1/ilya/locDoc/pyfst/june_models/bots_dist_gabor/ \
--preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/bots_dist_gabor/preprocessed.npz \
--st_type Botswana_dist_gabor \
--fst_preprocessing \
--network_spatial_size 1 \
--batch_size 1000

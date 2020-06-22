
# bots_dist_fst
# bots_sss_fst
# ip_dist_fst
# ip_sss_fst
# ksc_dist_fst
# ksc_sss_fst
# paviau_dist_fst
# paviau_sss_fst

# CUDA_VISIBLE_DEVICES=1 \
# python hyper_pixelNN.py \
# --fst_preprocessing \
# --st_type tang \
# --dataset IP \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/IP_strictsinglesite_trainval_s05_3_000392.mat \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/IP_wst \
# --preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/IP_wst/preprocessed.npz \
# --network_spatial_size 1 \
# --batch_size 1000 \
# --svm_predict

# CUDA_VISIBLE_DEVICES=1 \
# python hyper_pixelNN.py \
# --fst_preprocessing \
# --st_type tang \
# --dataset KSC \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/KSC_distributed_trainval_s50_4_112928.mat \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/KSC_wst \
# --preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/KSC_wst/preprocessed.npz \
# --network_spatial_size 1 \
# --batch_size 1000 \
# --svm_predict

# CUDA_VISIBLE_DEVICES=1 \
# python hyper_pixelNN.py \
# --fst_preprocessing \
# --st_type tang \
# --dataset Botswana \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/Botswana_distributed_trainval_s20_5_817361.mat \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/Botswana_wst \
# --preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/Botswana_wst/preprocessed.npz \
# --network_spatial_size 1 \
# --batch_size 1000 \
# --svm_predict

## conf mat

# CUDA_VISIBLE_DEVICES=0 \
# python hyper_pixelNN.py \
# --fst_preprocessing \
# --st_type tang \
# --dataset KSC \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/KSC_distributed_trainval_s20_9_760600.mat \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/KSC_wst \
# --preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/KSC_wst/preprocessed.npz \
# --network_spatial_size 1 \
# --batch_size 1000 \
# --svm_predict

CUDA_VISIBLE_DEVICES=0 \
python hyper_pixelNN.py \
--fst_preprocessing \
--st_type tang \
--dataset PaviaU \
--mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/PaviaU_distributed_trainval_s10_7_759773.mat \
--model_root /scratch0/ilya/locDoc/pyfst/models/pu_tang_svm \
--network_spatial_size 1 \
--batch_size 1000 \
--svm_predict
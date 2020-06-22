
# bots_dist_fst
# bots_sss_fst
# ip_dist_fst
# ip_sss_fst
# ksc_dist_fst
# ksc_sss_fst
# paviau_dist_fst
# paviau_sss_fst

# CUDA_VISIBLE_DEVICES=0 \
# python hyper_pixelNN.py \
# --dataset IP \
# --svm_predict \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/IP_distributed_trainval_s05_4_387762.mat \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/ip_dist_fst/ \
# --preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/ip_dist_fst/preprocessed.npz \
# --st_type IP_dist \
# --fst_preprocessing \
# --network_spatial_size 1 \
# --batch_size 1000

# CUDA_VISIBLE_DEVICES=0 \
# python hyper_pixelNN.py \
# --dataset IP \
# --svm_predict \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/IP_strictsinglesite_trainval_s05_8_351788.mat \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/ip_sss_fst/ \
# --preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/ip_sss_fst/preprocessed.npz \
# --st_type IP_SSS \
# --fst_preprocessing \
# --network_spatial_size 1 \
# --batch_size 1000

# CUDA_VISIBLE_DEVICES=0 \
# python hyper_pixelNN.py \
# --dataset KSC \
# --svm_predict \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/KSC_strictsinglesite_trainval_s50_4_514244.mat \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/ksc_sss_fst/ \
# --preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/ksc_sss_fst/preprocessed.npz \
# --st_type KSC_SSS \
# --fst_preprocessing \
# --network_spatial_size 1 \
# --batch_size 1000

# CUDA_VISIBLE_DEVICES=0 \
# python hyper_pixelNN.py \
# --dataset KSC \
# --svm_predict \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/KSC_distributed_trainval_s50_8_312496.mat \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/ksc_dist_fst/ \
# --preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/ksc_dist_fst/preprocessed.npz \
# --st_type KSC_dist \
# --fst_preprocessing \
# --network_spatial_size 1 \
# --batch_size 1000

# CUDA_VISIBLE_DEVICES=0 \
# python hyper_pixelNN.py \
# --dataset Botswana \
# --svm_predict \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/Botswana_strictsinglesite_trainval_s20_6_929435.mat \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/bots_sss_fst/ \
# --preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/bots_sss_fst/preprocessed.npz \
# --st_type Botswana_SSS \
# --fst_preprocessing \
# --network_spatial_size 1 \
# --batch_size 1000

# CUDA_VISIBLE_DEVICES=0 \
# python hyper_pixelNN.py \
# --dataset Botswana \
# --svm_predict \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/Botswana_distributed_trainval_s20_5_817361.mat \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/bots_dist_fst/ \
# --preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/bots_dist_fst/preprocessed.npz \
# --st_type Botswana_dist \
# --fst_preprocessing \
# --network_spatial_size 1 \
# --batch_size 1000

# CUDA_VISIBLE_DEVICES=0 \
# python hyper_pixelNN.py \
# --dataset PaviaU \
# --svm_predict \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/PaviaU_strictsinglesite_trainval_s90_2_291093.mat \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/paviau_sss_fst/ \
# --preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/paviau_sss_fst/preprocessed.npz \
# --st_type paviaU_SSS \
# --fst_preprocessing \
# --network_spatial_size 1 \
# --batch_size 1000

# CUDA_VISIBLE_DEVICES=0 \
# python hyper_pixelNN.py \
# --dataset PaviaU \
# --svm_predict \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/PaviaU_distributed_trainval_p0200_6_925171.mat \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/paviau_dist_fst/ \
# --preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/paviau_dist_fst/preprocessed.npz \
# --st_type paviaU_dist \
# --fst_preprocessing \
# --network_spatial_size 1 \
# --batch_size 1000

### FOR CONFUSION MATRICES

# CUDA_VISIBLE_DEVICES=0 \
# python hyper_pixelNN.py \
# --dataset IP \
# --svm_predict \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/IP_distributed_trainval_p0500_4_695632.mat \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/ip_dist_fst/ \
# --preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/ip_dist_fst/preprocessed.npz \
# --st_type IP_dist \
# --fst_preprocessing \
# --network_spatial_size 1 \
# --batch_size 1000

CUDA_VISIBLE_DEVICES=0 \
python hyper_pixelNN.py \
--dataset PaviaU \
--svm_predict \
--mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/PaviaU_distributed_trainval_s10_7_759773.mat \
--model_root /scratch1/ilya/locDoc/pyfst/june_models/paviau_dist_fst/ \
--preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/paviau_dist_fst/preprocessed.npz \
--st_type paviaU_dist \
--fst_preprocessing \
--network_spatial_size 1 \
--batch_size 1000

# CUDA_VISIBLE_DEVICES=0 \
# python hyper_pixelNN.py \
# --dataset KSC \
# --svm_predict \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/KSC_distributed_trainval_s20_8_029188.mat \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/ksc_dist_fst/ \
# --preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/ksc_dist_fst/preprocessed.npz \
# --st_type KSC_dist \
# --fst_preprocessing \
# --network_spatial_size 1 \
# --batch_size 1000

# CUDA_VISIBLE_DEVICES=0 \
# python hyper_pixelNN.py \
# --dataset Botswana \
# --svm_predict \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/Botswana_distributed_trainval_s10_5_552529.mat \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/bots_dist_fst/ \
# --preprocessed_data_path /scratch1/ilya/locDoc/pyfst/june_models/bots_dist_fst/preprocessed.npz \
# --st_type Botswana_dist \
# --fst_preprocessing \
# --network_spatial_size 1 \
# --batch_size 1000
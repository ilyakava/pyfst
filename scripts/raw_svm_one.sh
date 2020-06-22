
# Botswana_raw_svm
# IP_raw_svm
# KSC_raw_svm

# CUDA_VISIBLE_DEVICES= \
# python hyper_pixelNN.py \
# --dataset IP \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/IP_strictsinglesite_trainval_s05_5_218470.mat \
# --model_root /scratch0/ilya/locDoc/pyfst/june_models/IP_raw_svm \
# --network_spatial_size 1 \
# --batch_size 1000 \
# --svm_predict

# CUDA_VISIBLE_DEVICES= \
# python hyper_pixelNN.py \
# --dataset KSC \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/KSC_distributed_trainval_s50_2_630096.mat \
# --model_root /scratch0/ilya/locDoc/pyfst/june_models/KSC_raw_svm \
# --network_spatial_size 1 \
# --batch_size 1000 \
# --svm_predict

CUDA_VISIBLE_DEVICES= \
python hyper_pixelNN.py \
--dataset Botswana \
--mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/Botswana_distributed_trainval_s20_1_935671.mat \
--model_root /scratch0/ilya/locDoc/pyfst/june_models/Botswana_raw_svm \
--network_spatial_size 1 \
--batch_size 1000 \
--svm_predict

# CUDA_VISIBLE_DEVICES= \
# python hyper_pixelNN.py \
# --dataset PaviaU \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/PaviaU_strictsinglesite_trainval_s90_2_291093.mat \
# --model_root /scratch0/ilya/locDoc/pyfst/models/pu_raw_svm \
# --network_spatial_size 1 \
# --batch_size 1000 \
# --svm_predict


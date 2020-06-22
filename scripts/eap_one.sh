


# CUDA_VISIBLE_DEVICES=1 \
# python hyper_pixelNN.py \
# --dataset IP \
# --npca_components 4 \
# --attribute_profile \
# --batch_size 50 \
# --lr 0.00001 \
# --network aptoula \
# --network_spatial_size 9 \
# --eval_period 25 \
# --num_epochs 100000 \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/IP_strictsinglesite_trainval_s05_4_141968.mat \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/eap/ip/eap/IP_strictsinglesite_trainval_s05_4_141968/0/ \
# --predict

# CUDA_VISIBLE_DEVICES=0 \
# python hyper_pixelNN.py \
# --dataset KSC \
# --npca_components 4 \
# --attribute_profile \
# --batch_size 50 \
# --lr 0.0001 \
# --network aptoula \
# --network_spatial_size 9 \
# --eval_period 2 \
# --num_epochs 100000 \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/KSC_distributed_trainval_s50_9_366224.mat \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/eap/eap/KSC_distributed_trainval_s50_9_366224/0/ \
# --predict

# CUDA_VISIBLE_DEVICES=0 \
# python hyper_pixelNN.py \
# --dataset Botswana \
# --npca_components 4 \
# --attribute_profile \
# --batch_size 50 \
# --lr 0.0001 \
# --network aptoula \
# --network_spatial_size 9 \
# --eval_period 2 \
# --num_epochs 100000 \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/Botswana_distributed_trainval_s20_2_145243.mat \
# --model_root /scratch0/ilya/locDoc/pyfst/june_models/eap/Botswana_distributed_trainval_s20_2_145243/0/ \
# --predict

## conf mat



CUDA_VISIBLE_DEVICES=1 \
python hyper_pixelNN.py \
--dataset PaviaU \
--npca_components 4 \
--attribute_profile \
--batch_size 50 \
--lr 0.00005 \
--network aptoula \
--network_spatial_size 9 \
--eval_period 25 \
--num_epochs 100000 \
--mask_root  /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/PaviaU_distributed_trainval_p0100_6_324571.mat \
--model_root /scratch0/ilya/locDoc/pyfst/models/eap/PaviaU_distributed_trainval_p0100_6_324571/0/ \
--predict



# CUDA_VISIBLE_DEVICES=1 \
# python hyper_pixelNN.py \
# --dataset IP \
# --npca_components 3 \
# --batch_size 100 \
# --lr 0.00005 \
# --network DFFN_3tower_4depth \
# --network_spatial_size 25 \
# --eval_period 5 \
# --num_epochs 100000 \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/IP_strictsinglesite_trainval_s05_2_028792.mat \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/dffn/ip/dffn/IP_strictsinglesite_trainval_s05_2_028792/0/ \
# --predict

# CUDA_VISIBLE_DEVICES=1 \
# python hyper_pixelNN.py \
# --dataset KSC \
# --npca_components 5 \
# --batch_size 100 \
# --lr 0.0005 \
# --network DFFN_3tower_4depth \
# --network_spatial_size 25 \
# --eval_period 2 \
# --num_epochs 100000 \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/KSC_distributed_trainval_s05_9_084056.mat \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/dffn/dffn/KSC_distributed_trainval_s50_9_366224/0/ \
# --predict

# CUDA_VISIBLE_DEVICES=1 \
# python hyper_pixelNN.py \
# --dataset Botswana \
# --npca_components 5 \
# --batch_size 100 \
# --lr 0.0005 \
# --network DFFN_3tower_4depth \
# --network_spatial_size 25 \
# --eval_period 2 \
# --num_epochs 100000 \
# --mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/Botswana_distributed_trainval_s20_6_315569.mat \
# --model_root /scratch1/ilya/locDoc/pyfst/june_models/dffn/dffn/Botswana_distributed_trainval_s20_6_315569/0/ \
# --predict


### Conf mat


CUDA_VISIBLE_DEVICES=1 \
python hyper_pixelNN.py \
--dataset IP \
--npca_components 3 \
--batch_size 100 \
--lr 0.00005 \
--network DFFN_3tower_4depth \
--network_spatial_size 25 \
--eval_period 5 \
--num_epochs 100000 \
--mask_root /cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/IP_distributed_trainval_p0500_8_250034.mat \
--model_root /scratch0/ilya/locDoc/pyfst/models/dffn/IP_distributed_trainval_p0500_8_250034/0/ \
--predict
CUDA_VISIBLE_DEVICES= \
python hyper_pixelNN.py \
--dataset KSC \
--svm_predict \
--mask_root ./masks/KSC_strictsinglesite_trainval_s50_8_191432.mat \
--model_root /scratch0/ilya/locDoc/pyfst/june_models/ksc_sss_raw_svm \
--network_spatial_size 1 \
--batch_size 1000

CUDA_VISIBLE_DEVICES=0 \
python hyper_pixelNN.py \
--dataset KSC \
--svm_predict \
--mask_root ./masks/KSC_strictsinglesite_trainval_s50_8_191432.mat \
--model_root /scratch0/ilya/locDoc/pyfst/june_models/ksc_sss_gabor_svm \
--preprocessed_data_path /scratch0/ilya/locDoc/pyfst/june_models/ksc_sss_gabor_svm/preprocessed.npz \
--st_type KSC_gabor \
--fst_preprocessing \
--network_spatial_size 1 \
--batch_size 1000

# --preprocessed_data_path /scratch0/ilya/locDoc/pyfst/models/ksc_svm/preprocessed.npz \
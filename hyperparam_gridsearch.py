"""Iterate through hyperparams of ST to get accuracies
"""

import os
import time

import numpy as np
from sklearn.svm import SVC
from tqdm import tqdm

import hyper_pixelNN as hsinn

import pdb

FEAT_DIR = '/scratch0/ilya/locDoc/data/hyperspec/features/npz_feat'
FEAT_DIR = '/scratch1/ilya/locDoc/data/hyperspec/features/npz_feat'
FEAT_DIR = '/scratch2/ilyak/locDoc/data/hyperspec/features/npz_feat'

RESULTS_DIR = '/scratch0/ilya/locDoc/pyfst'
RESULTS_DIR = '/scratch2/ilyak/locDoc/pyfst'

def run_svm(trainX, trainY, testX, testY):
    print('starting training')
    start = time.time()
    clf = SVC(kernel='linear', C=1000.0)
    clf.fit(trainX, trainY)
    end = time.time()
    print('Training done. Took %is' % int(end - start))

    # now test
    test_chunk_size = 1000

    # acc = clf.score(testX, testY)
    # better to score in batches to limit memory use:
    n_correct = 0
    for i in tqdm(range(0,testY.shape[0],test_chunk_size), desc='Testing'):
        p_label = clf.predict(testX[i:i+test_chunk_size]);
        n_correct += (p_label == testY[i:i+test_chunk_size]).sum()
    acc = float(n_correct) / testY.shape[0]
    return acc
        

def get_acc_for_config(dataset, st_net_spec, preprocessed_data_path, preprocessed_data_root, mask_paths, save_features):
    """One set of hyperparams and dataset, several trials
    """
    trainimgname, trainlabelname = hsinn.dset_filenames_dict[dataset]
    trainimgfield, trainlabelfield = hsinn.dset_fieldnames_dict[dataset]
    data, labels = hsinn.load_data(trainimgname, trainimgfield, trainlabelname, trainlabelfield)
    
    if save_features:
        data = hsinn.load_or_preprocess_data(data, preprocessed_data_path, preprocessed_data_root, st_net_spec, 31)
    else:
        data = hsinn.preprocess_data(data, st_net_spec, 31)
    
    accs = []
    # loop through masks
    for mask_path in mask_paths:
        train_mask = hsinn.multiversion_matfile_get_field(mask_path, 'train_mask')
        val_mask = hsinn.multiversion_matfile_get_field(mask_path, 'test_mask')
        trainX, trainY, valX, valY = hsinn.get_train_val_splits(data, labels, train_mask, val_mask, (0,0,0)) # , n_eval=2048
    
        # run svm
        accs.append(run_svm(trainX.squeeze(), trainY, valX.squeeze(), valY))
    print('[%s] Avg acc is %.2f' % (hsinn.spec_to_str(st_net_spec), sum(accs) / len(accs)))
    return accs
    

def specs1():
    """
    Spatial psis should be decreasing, phi should be equal to smallest psi
    """
    specs = []
    myrange = [9,7,5,3]
    for b in myrange:
        for s1 in myrange:
            for s2 in myrange:
                if s1 >= s2:
                    specs.append( [ [b,s1,s1], [b,s2,s2], [b,s2,s2] ] )
    return specs
    
def specs2():
    """
    Spatial psis should be decreasing, phi should be equal to smallest psi
    """
    specs = []
    myrange = [7,5,3,1]
    for b in [9,7,5,3]:
        for s1 in myrange:
            for s2 in myrange:
                if s1 >= s2:
                    specs.append( [ [b,s1,s1], [b,s2,s2], [b,s2,s2] ] )
    return specs
    
def specs_append1():
    """
    Spatial psis should be decreasing, phi should be equal to smallest psi
    """
    specs = []
    myrange = [9,7,5,3]
    for b in myrange:
        for s1 in [1]:
            for s2 in [1]:
                if s1 >= s2:
                    specs.append( [ [b,s1,s1], [b,s2,s2], [b,s2,s2] ] )
    return specs
    
def specs_append9():
    """
    Spatial psis should be decreasing, phi should be equal to smallest psi
    """
    specs = []
    myrange = [9,7,5,3]
    for b in myrange:
        for s1 in [9]:
            for s2 in myrange:
                if s1 >= s2:
                    specs.append( [ [b,s1,s1], [b,s2,s2], [b,s2,s2] ] )
    return specs
    
def specs_append9_bands():
    """
    Spatial psis should be decreasing, phi should be equal to smallest psi
    """
    specs = []
    myrange = [7,5,3]
    for b in [9]:
        for s1 in myrange:
            for s2 in myrange:
                if s1 >= s2:
                    specs.append( [ [b,s1,s1], [b,s2,s2], [b,s2,s2] ] )
    return specs
    
def perform_gridsearch(dataset, masks, specs, outfile='gridsearch.npz', save_features=False, preprocessed_data_root=FEAT_DIR):
    # a list< list < int > > -> list< float >.
    # Maps a configuration to a set of accuracies
    results = {}
    for spec_i, spec in enumerate(specs):
        psi1,psi2,phi = spec
    
        st_net_spec = hsinn.st_net_spec_struct(psi1,psi2,phi)
    
        preprocessed_data_path = os.path.join(preprocessed_data_root, '%s__%s.npz' % (dataset, hsinn.spec_to_str(st_net_spec)))
    
        accs = get_acc_for_config(dataset, st_net_spec, preprocessed_data_path, preprocessed_data_root, masks, save_features)
        results[hsinn.spec_to_str(st_net_spec)] = accs
        print('FINISHED %i/%i' % (spec_i+1, len(specs)))
    
        np.savez(os.path.join(RESULTS_DIR, outfile), results=results)
    print('Saved %s' % os.path.join(RESULTS_DIR, outfile))
    

def paviaU():
    dataset = 'PaviaU'
    # masks = ['PaviaU_gt_traintest_s03_1_3f6384.mat',
    #          'PaviaU_gt_traintest_s03_2_b67b5f.mat',
    #          'PaviaU_gt_traintest_s03_3_7d8356.mat',
    #          'PaviaU_gt_traintest_s03_4_241266.mat',
    #         'PaviaU_gt_traintest_s03_5_ccbbb1.mat',
    #         'PaviaU_gt_traintest_s03_6_dce186.mat',
    #         'PaviaU_gt_traintest_s03_7_d5cdfe.mat',
    #         'PaviaU_gt_traintest_s03_8_6bcd5a.mat',
    #         'PaviaU_gt_traintest_s03_9_a1ff2b.mat',
    #         'PaviaU_gt_traintest_s03_10_e1dac2.mat']
    
             
#     masks = ['PaviaU_strictsinglesite_trainval_s90_0_700083.mat',
# 'PaviaU_strictsinglesite_trainval_s90_1_509297.mat',
# 'PaviaU_strictsinglesite_trainval_s90_2_291093.mat',
# 'PaviaU_strictsinglesite_trainval_s90_3_232341.mat',
# 'PaviaU_strictsinglesite_trainval_s90_4_833337.mat',
# 'PaviaU_strictsinglesite_trainval_s90_5_672053.mat',
# 'PaviaU_strictsinglesite_trainval_s90_6_215589.mat',
# 'PaviaU_strictsinglesite_trainval_s90_7_553699.mat',
# 'PaviaU_strictsinglesite_trainval_s90_8_734449.mat',
# 'PaviaU_strictsinglesite_trainval_s90_9_484519.mat']

    masks = ['PaviaU_distributed_trainval_p0200_0_112681.mat',
'PaviaU_distributed_trainval_p0200_1_678193.mat',
'PaviaU_distributed_trainval_p0200_2_123909.mat',
'PaviaU_distributed_trainval_p0200_3_346269.mat',
'PaviaU_distributed_trainval_p0200_4_987781.mat',
'PaviaU_distributed_trainval_p0200_5_057029.mat',
'PaviaU_distributed_trainval_p0200_6_925171.mat',
'PaviaU_distributed_trainval_p0200_7_971587.mat',
'PaviaU_distributed_trainval_p0200_8_776217.mat',
'PaviaU_distributed_trainval_p0200_9_216285.mat']
    
    # masks = [os.path.join('/scratch0/ilya/locDoc/data/hyperspec/',m) for m in masks]
    # perform_gridsearch(dataset, masks, specs1(), outfile='gridsearch_paviaU_s200_10trials_full_new.npz', save_features=True)
#     masks = ['PaviaU_strictsinglesite_trainval_s50_0_979087.mat',
# 'PaviaU_strictsinglesite_trainval_s50_1_270161.mat',
# 'PaviaU_strictsinglesite_trainval_s50_2_259715.mat',
# 'PaviaU_strictsinglesite_trainval_s50_3_107251.mat',
# 'PaviaU_strictsinglesite_trainval_s50_4_729473.mat',
# 'PaviaU_strictsinglesite_trainval_s50_5_129325.mat',
# 'PaviaU_strictsinglesite_trainval_s50_6_927653.mat',
# 'PaviaU_strictsinglesite_trainval_s50_7_627051.mat',
# 'PaviaU_strictsinglesite_trainval_s50_8_525881.mat',
# 'PaviaU_strictsinglesite_trainval_s50_9_785489.mat']


    masks = [os.path.join('/cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/',m) for m in masks]
    perform_gridsearch(dataset, masks, specs1(), outfile='gridsearch_june19_paviaU_distributed_p0200_10trials.npz', save_features=True)

def Botswana():
    dataset = 'Botswana'
#     masks = ['Botswana_strictsinglesite_trainval_s20_0_840407.mat',
# 'Botswana_strictsinglesite_trainval_s20_1_027359.mat',
# 'Botswana_strictsinglesite_trainval_s20_2_059593.mat',
# 'Botswana_strictsinglesite_trainval_s20_3_800757.mat',
# 'Botswana_strictsinglesite_trainval_s20_4_848729.mat',
# 'Botswana_strictsinglesite_trainval_s20_5_309369.mat',
# 'Botswana_strictsinglesite_trainval_s20_6_913005.mat',
# 'Botswana_strictsinglesite_trainval_s20_7_508879.mat',
# 'Botswana_strictsinglesite_trainval_s20_8_218687.mat',
# 'Botswana_strictsinglesite_trainval_s20_9_288573.mat']

#     masks = ['Botswana_strictsinglesite_trainval_s20_0_619497.mat',
# 'Botswana_strictsinglesite_trainval_s20_1_371565.mat',
# 'Botswana_strictsinglesite_trainval_s20_2_133689.mat',
# 'Botswana_strictsinglesite_trainval_s20_3_578661.mat',
# 'Botswana_strictsinglesite_trainval_s20_4_954915.mat',
# 'Botswana_strictsinglesite_trainval_s20_5_406769.mat',
# 'Botswana_strictsinglesite_trainval_s20_6_929435.mat',
# 'Botswana_strictsinglesite_trainval_s20_7_053419.mat',
# 'Botswana_strictsinglesite_trainval_s20_8_379701.mat',
# 'Botswana_strictsinglesite_trainval_s20_9_900903.mat']

    masks = ['Botswana_distributed_trainval_s20_0_383283.mat',
'Botswana_distributed_trainval_s20_1_935671.mat',
'Botswana_distributed_trainval_s20_2_145243.mat',
'Botswana_distributed_trainval_s20_3_753761.mat',
'Botswana_distributed_trainval_s20_4_716839.mat',
'Botswana_distributed_trainval_s20_5_817361.mat',
'Botswana_distributed_trainval_s20_6_315569.mat',
'Botswana_distributed_trainval_s20_7_737133.mat',
'Botswana_distributed_trainval_s20_8_637855.mat',
'Botswana_distributed_trainval_s20_9_289575.mat']

    masks = [os.path.join('/cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/',m) for m in masks]
    perform_gridsearch(dataset, masks, specs1(), outfile='gridsearch_june19_Bots_distributed_s20_10trials.npz', save_features=True)

def KSC():
    dataset = 'KSC'
#     masks = ['KSC_strictsinglesite_trainval_s50_0_665848.mat',
# 'KSC_strictsinglesite_trainval_s50_1_783796.mat',
# 'KSC_strictsinglesite_trainval_s50_2_455308.mat',
# 'KSC_strictsinglesite_trainval_s50_3_643924.mat',
# 'KSC_strictsinglesite_trainval_s50_4_514244.mat',
# 'KSC_strictsinglesite_trainval_s50_5_762052.mat',
# 'KSC_strictsinglesite_trainval_s50_6_668196.mat',
# 'KSC_strictsinglesite_trainval_s50_7_990120.mat',
# 'KSC_strictsinglesite_trainval_s50_8_191432.mat',
# 'KSC_strictsinglesite_trainval_s50_9_307364.mat']

    masks = ['KSC_distributed_trainval_s20_0_128852.mat',
'KSC_distributed_trainval_s20_1_644908.mat',
'KSC_distributed_trainval_s20_2_182432.mat',
'KSC_distributed_trainval_s20_3_100036.mat',
'KSC_distributed_trainval_s20_4_353740.mat',
'KSC_distributed_trainval_s20_5_348604.mat',
'KSC_distributed_trainval_s20_6_263496.mat',
'KSC_distributed_trainval_s20_7_751704.mat',
'KSC_distributed_trainval_s20_8_029188.mat',
'KSC_distributed_trainval_s20_9_760600.mat']

    masks = [os.path.join('/cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/', m) for m in masks]
    perform_gridsearch(dataset, masks, specs1(), outfile='gridsearch_june19_KSC_distributed_s20_10trials.npz', save_features=True)

def IP():
    dataset = 'IP'
#     masks = ['IP_strictsinglesite_trainval_s20_0_211328.mat',
# 'IP_strictsinglesite_trainval_s20_1_881136.mat',
# 'IP_strictsinglesite_trainval_s20_2_869530.mat',
# 'IP_strictsinglesite_trainval_s20_3_014114.mat',
# 'IP_strictsinglesite_trainval_s20_4_586540.mat',
# 'IP_strictsinglesite_trainval_s20_5_805378.mat',
# 'IP_strictsinglesite_trainval_s20_6_666022.mat',
# 'IP_strictsinglesite_trainval_s20_7_581782.mat',
# 'IP_strictsinglesite_trainval_s20_8_901868.mat',
# 'IP_strictsinglesite_trainval_s20_9_778852.mat']

#     masks = ['IP_strictsinglesite_trainval_p2000_0_533076.mat',
# 'IP_strictsinglesite_trainval_p2000_1_494066.mat',
# 'IP_strictsinglesite_trainval_p2000_2_984650.mat',
# 'IP_strictsinglesite_trainval_p2000_3_340434.mat',
# 'IP_strictsinglesite_trainval_p2000_4_605220.mat',
# 'IP_strictsinglesite_trainval_p2000_5_199324.mat',
# 'IP_strictsinglesite_trainval_p2000_6_596068.mat',
# 'IP_strictsinglesite_trainval_p2000_7_199664.mat',
# 'IP_strictsinglesite_trainval_p2000_8_881256.mat',
# 'IP_strictsinglesite_trainval_p2000_9_005288.mat']

    masks = ['IP_distributed_trainval_p1000_0_970730.mat',
'IP_distributed_trainval_p1000_1_532502.mat',
'IP_distributed_trainval_p1000_2_339308.mat',
'IP_distributed_trainval_p1000_3_756872.mat',
'IP_distributed_trainval_p1000_4_633060.mat',
'IP_distributed_trainval_p1000_5_914454.mat',
'IP_distributed_trainval_p1000_6_667722.mat',
'IP_distributed_trainval_p1000_7_240942.mat',
'IP_distributed_trainval_p1000_8_272914.mat',
'IP_distributed_trainval_p1000_9_521992.mat']

    masks = [os.path.join('/cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/', m) for m in masks]
    perform_gridsearch(dataset, masks, specs1(), outfile='gridsearch_june19b_IP_distributed_p10_10trials.npz', save_features=True)

    masks = ['IP_strictsinglesite_trainval_p1000_0_766900.mat',
'IP_strictsinglesite_trainval_p1000_1_413194.mat',
'IP_strictsinglesite_trainval_p1000_2_252770.mat',
'IP_strictsinglesite_trainval_p1000_3_582176.mat',
'IP_strictsinglesite_trainval_p1000_4_834150.mat',
'IP_strictsinglesite_trainval_p1000_5_599784.mat',
'IP_strictsinglesite_trainval_p1000_6_076514.mat',
'IP_strictsinglesite_trainval_p1000_7_168940.mat',
'IP_strictsinglesite_trainval_p1000_8_531590.mat',
'IP_strictsinglesite_trainval_p1000_9_613356.mat']

    masks = [os.path.join('/cfarhomes/ilyak/ilyakavalerov@gmail.com/ramawks69/pyfst/masks/', m) for m in masks]
    perform_gridsearch(dataset, masks, specs2(), outfile='gridsearch_june19_IP_strictsinglesite_p10_10trials.npz', save_features=True)



def main():
    # paviaU()
    # IP()
    KSC()
    
    # Botswana()

if __name__ == '__main__':
    main()
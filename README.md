# Scattering Transforms in Python

Please cite:

```
@ARTICLE{3dfst,
  author={I. {Kavalerov} and W. {Li} and W. {Czaja} and R. {Chellappa}},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={3-D Fourier Scattering Transform and Classification of Hyperspectral Images}, 
  year={2020},
  volume={},
  number={},
  pages={1-16},
  doi={10.1109/TGRS.2020.3040203}}
```

Based on the [arxiv](https://arxiv.org/pdf/1906.06804.pdf), [ieee](https://ieeexplore.ieee.org/document/9288871) papers.

### Usage

The main file for HSI analysis is `hyper_pixelNN.py`.

To use this file run the following scripts:

- `sh scripts/final/train_dffn_IP_replicate.sh`, Replicates the results of DFFN for Indian Pines once.
- `sh scripts/final/train_EAP_PaviaU_replicate.sh`, Replicates the results of EAP-Area for PaviaU once.
- `python scripts/generate_many_dl_runs.py`, Generates scripts that will run 10 trials of DFFN and EAP for every dataset.
- `scripts/final/IP_fst_svm_all.sh`, runs FST and a SVM once for every mask listed in a txt file.
- `scripts/final/svm_wst_all.sh`, runs WST and SVMs for the 4 datasets mentioned in the paper.
- `scripts/final/svm_wst_all.sh`, runs SVMs on the raw HSI images for the 4 datasets mentioned in the paper.`

## Installation for scripts in scripts/final

- install items in "Software Versioning" below
- compile libsvm in `lib/libsvm` (run `make` in `lib/libsvm` and in `lib/libsvm/python`)
- get masks
- get mask lists
- get data

Archived version with masks is [here](https://github.com/ilyakava/pyfst/releases/tag/MajRev1).

### Downloading Data

You may download a copy of the HSI data mentioned in the paper [here](https://drive.google.com/file/d/1u6fzTztudcilKUmV9ZKUh6khZTIeAeB7/view?usp=sharing)

Or, see [the GIC website](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) and download the datasets there.

### Training Masks

You may download the exact training masks we use in the paper [get the masks inside the release here](https://github.com/ilyakava/pyfst/releases/tag/MajRev1) and place them in your `masks` directory.

When using options like `--svm_multi_mask_file_list` there should be a txt file that lists the fully qualified path to each mask file that should be used.

#### Create Custom Training/Testing Splits

Create more training/testing splits with `sites_train_val_split.py`.

## Software Versioning

Tested on Python 2.7.14 (Anaconda), tensorflow 1.10.1, cuda 9.0.176, cudnn-7.0 (8.0 might work too). Red Hat Enterprise Linux Workstation release 7.6 (Maipo). GeForce GTX TITAN X.

```
conda create -n venvtfnonb python=2.7.14
pip install --ignore-installed --upgrade tensorflow-gpu==1.10.1
pip install sklearn tqdm h5py hdf5storage pillow matplotlib plotly scikit-image
```

## Docker instructions

This docker container was tested on a Nvidia 3090 and a M40.

### Setup from scratch

```
cd ~
git clone https://github.com/ilyakava/pyfst.git
mkdir -p hsi_data/derived
```

```
docker pull nvcr.io/nvidia/tensorflow:18.10-py2
docker run --gpus all --shm-size 16G --rm -it -v $HOME/pyfst:/pyfst -v $HOME/hsi_data:/data nvcr.io/nvidia/tensorflow:18.10-py2
pip install sklearn tqdm h5py hdf5storage pillow matplotlib plotly scikit-image
cd /pyfst/lib/libsvm
make
cd python
make
cd /pyfst/masks
wget https://github.com/ilyakava/pyfst/releases/download/MajRev1/pyfst.tar.gz
tar -zxvf pyfst.tar.gz
rm pyfst.tar.gz
mv pyfst/masks/* ./
rm -rf pyfst
cd /pyfst
apt-get update
apt-get install python-tk
```

### Run an example

```
export DATASET_PATH=/data
export DATA_PATH=/data/derived
cd /pyfst/
CUDA_VISIBLE_DEVICES=0 sh scripts/KSC_fst_svm_one.sh
```

For this example the ST runs on the M40 in 110s.
For the newer 3090, the Ampere architecture requires more graph compilation time because of the older cuda version. 

## License

MIT

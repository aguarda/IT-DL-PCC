# Instituto de Telecomunicações Deep Learning-based Point Cloud Codec (IT-DL-PCC)

This is the software of the codecs submitted to the Call for Proposals on JPEG Pleno Point Cloud Coding issued on January 2022:
    - IT-DL-PCC-G: Geometry-only codec
    - IT-DL-PCC-GC: Joint geometry + color codec

These proposals have been originated by research developed at Instituto de Telecomunicações (IT), in the context of the project Deep-PCR entitled “Deep learning-based Point Cloud Representation” (PTDC/EEI-COM/1125/2021), financed by Fundação para a Ciência e Tecnologia (FCT).

* Authors: [André F. R. Guarda](https://scholar.google.com/citations?user=GqwCCpYAAAAJ)<sup>1</sup>, [Nuno M. M. Rodrigues](https://scholar.google.com/citations?user=UOIzJ50AAAAJ)<sup>2</sup>, Manuel Ruivo<sup>1</sup>, Luís Coelho<sup>1</sup>, Abdelrahman Seleem<sup>1</sup>, [Fernando Pereira](https://scholar.google.com/citations?user=ivtyoBcAAAAJ)<sup>1</sup>
* Affiliations:
  * <sup>1</sup> Instituto Superior Técnico - Universidade de Lisboa, and Instituto de Telecomunicações, Lisbon, Portugal
  * <sup>2</sup> ESTG, Politécnico de Leiria and Instituto de Telecomunicações, Leiria, Portugal

# Contents

* **Source code**: The full source code in `src` to run and train both the IT-DL-PCC codecs.
* **Rate-Distortion performance results**: The full results obtained for all test point clouds of the JPEG Pleno Point Cloud Coding Call for Proposals are included in `results`.
* **Trained DL models**: All the trained DL coding models used in the codec can be downloaded [here](https://drive.google.com/file/d/1JDkvcfVcqlwP6TW8Hldw-LS00cRYVsp8/view?usp=sharing). For both geometry-only or joint geometry + color coding, there are 6 DL coding models trained for different rate-distortion trade-offs (Geometry-only: λ = 0.00025, 0.0005, 0.001, 0.0025, 0.005 and 0.01; Joint Geometry + Color: λ = 0.000125, 0.00025, 0.0005, 0.001, 0.002 and 0.004), and 2 DL super-resolution models trained for different sampling factors (2 and 4), for a total of 16 DL models.

# Requirements

The prerequisites to run the IT-DL-PCC software, and the DL coding models in particular, are:

*	Python 3.9.7
*	PyTorch 1.12 with CUDA Version 11.6, or compatible
*	[CompressAI 1.2](https://github.com/InterDigitalInc/CompressAI)
*	TensorBoard 2.9
*	Python packages:
	*	numpy
	*	scipy
	*	scikit-learn
	*	pandas
	*	pyntcloud
	*	torchsummary

Using a Linux distribution (e.g. Ubuntu) is recommended.

# Usage

The main script `IT-DL-PCC.py` is used to encode and decode a PC using the IT-DL-PCC codec. The `train.py` and `train_sr.py` scripts are used to train the DL coding model and the DL super-resolution model, respectively.

## Running the main script:
```
python IT-DL-PCC.py [--with_color] [--cuda] {compress,decompress} [OPTIONS]
```

The flag `--with_color`, used before the `{compress,decompress}` command, should be given to select joint geometry + color coding. Otherwise, geometry-only coding is performed instead.

The flag `--cuda`, used before the `{compress,decompress}` command, should be given to run on GPU. Otherwise, the codec will run on CPU only.

## Encoding a Point Cloud:
```
usage: IT-DL-PCC.py [--with_color] [--cuda] compress [-h] [--helpfull] [--blk_size BLK_SIZE] [--q_step Q_STEP] [--scale SCALE] [--topk_metrics TOPK_METRICS] [--color_weight COLOR_WEIGHT] [--use_fast_topk] [--max_topk MAX_TOPK] [--topk_patience TOPK_PATIENCE] [--use_sr] [--sr_model_dir SR_MODEL_DIR] [--sr_topk SR_TOPK] [--sr_max_topk SR_MAX_TOPK] input_file model_dir output_dir

Reads a Point Cloud PLY file, compresses it, and writes the bitstream file.

positional arguments:
  input_file            Input Point Cloud filename (.ply).
  model_dir             Directory where to load model checkpoints. For compression, a single directory should be provided: ../models/test/checkpoint_best_loss.pth.tar
  output_dir            Directory where the compressed Point Cloud will be saved.

optional arguments:
  -h, --help            show this help message and exit
  --helpfull            show full help message and exit
  --blk_size BLK_SIZE   Size of the 3D coding block units. (default: 128)
  --q_step Q_STEP       Explicit quantization step. (default: 1)
  --scale SCALE         Down-sampling scale. If 'None', it is automatically determined. (default: None)
  --topk_metrics TOPK_METRICS
                        Metrics to use for the optimized Top-k binarization. Available: 'd1', 'd2', 'd1yuv', 'd2yuv', 'd1rgb', 'd2rgb'. If coding geometry-only, only the geometry metric is used. (default:
                        d1yuv)
  --color_weight COLOR_WEIGHT
                        Weight of the color metric in the joint top-k optimization. Between 0 and 1. (default: 0.5)
  --use_fast_topk       Use faster top-k optimization algorithm for the coding model. (default: False)
  --max_topk MAX_TOPK   Define the maximum factor used by the Top-K optimization algorithms. (default: 10)
  --topk_patience TOPK_PATIENCE
                        Define the patience for early stopping in the Top-K optimization algorithms. (default: 5)
  --use_sr              Use basic Upsampling (False) or Upsampling with Super-Resolution - SR (True). (default: False)
  --sr_model_dir SR_MODEL_DIR
                        Directory where to load Super-Resolution model. A single directory should be provided for the target down-sampling scale: ../models/test/checkpoint_best_loss.pth.tar (default: )
  --sr_topk SR_TOPK     Type of Top-k optimization for the Super-Resolution at the encoder: 'none' - Use the same as the coding model. 'full' - Use the regular algorithm (default). 'fast' - Use the faster algorithm.
                        (default: full)
  --sr_max_topk SR_MAX_TOPK
                        Define the maximum factor used by the Top-K optimization algorithms. (default: 10)

```

Usage examples:
```
python IT-DL-PCC.py compress "../test_data_path/longdress.ply" "../models/Geo/Codec/0.00025/checkpoint_best_loss.pth.tar" "../results/G/0.00025"
```
```
python IT-DL-PCC.py --with_color compress "../test_data_path/longdress.ply" "../models/Joint/Codec/0.00025/checkpoint_best_loss.pth.tar" "../results/GC/0.00025"
```
```
python IT-DL-PCC.py --with_color compress "../test_data_path/longdress.ply" "../models/Joint/Codec/0.00025/checkpoint_best_loss.pth.tar" "../results/GC/0.00025" --blk_size 64 --scale 2 --topk_metrics d2rgb
```
```
python IT-DL-PCC.py compress "../test_data_path/longdress.ply" "../models/Geo/Codec/0.0025/checkpoint_best_loss.pth.tar" "../results/G/0.0025" --scale 4 --use_sr --sr_model_dir "../models/Geo/SR/SF_4/checkpoint_best_loss.pth.tar"
```

## Decoding a point cloud:
```
usage: IT-DL-PCC.py [--with_color] [--cuda] decompress [-h] [--helpfull] [--sr_model_dir SR_MODEL_DIR] input_file model_dir

Reads a bitstream file, decodes the voxel blocks, and reconstructs the Point Cloud.

positional arguments:
  input_file            Input bitstream filename (.gz).
  model_dir             Directory where to load model checkpoints. For decompression, a single directory should be provided: ../models/test/checkpoint_best_loss.pth.tar

optional arguments:
  -h, --help            show this help message and exit
  --helpfull            show full help message and exit
  --sr_model_dir SR_MODEL_DIR
                        Directory where to load Super-Resolution model. A single directory should be provided for the target down-sampling scale: ../models/test/checkpoint_best_loss.pth.tar (default: )
```

Usage examples:
```
python IT-DL-PCC.py decompress "../results/G/0.00025/longdress/longdress.gz" "../models/Geo/Codec/0.00025/checkpoint_best_loss.pth.tar"
```
```
python IT-DL-PCC.py --with_color decompress "../results/GC/0.00025/longdress/longdress.gz" "../models/Joint/Codec/0.00025/checkpoint_best_loss.pth.tar"
```
```
python IT-DL-PCC.py decompress "../results/G/0.0025/longdress/longdress.gz" "../models/Geo/Codec/0.0025/checkpoint_best_loss.pth.tar" --sr_model_dir "../models/Geo/SR/SF_4/checkpoint_best_loss.pth.tar"
```

## Training a DL coding model:
```
usage: train.py [-h] [-v] [--with_color] -d TRAIN_DATA --val_data VAL_DATA --model_dir MODEL_DIR [-e EPOCHS] [-n NUM_WORKERS] [--batch-size BATCH_SIZE] [--cuda] [--save] [--seed SEED]
                [--clip_max_norm CLIP_MAX_NORM] [--checkpoint CHECKPOINT] [--logs LOGS] [--num_filters NUM_FILTERS] [--lambda LMBDA] [--fl_alpha FL_ALPHA] [--fl_gamma FL_GAMMA] [--omega OMEGA]

Codec training script.

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         Print model summary.
  --with_color          Jointly compress both geometry and color.
  -d TRAIN_DATA, --train_data TRAIN_DATA
                        Path to training PC data (folder with pickle .pkl files).
  --val_data VAL_DATA   Path to validation PC data (folder with pickle .pkl files).
  --model_dir MODEL_DIR
                        Directory where to save the model checkpoints.
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs (default: 500)
  -n NUM_WORKERS, --num-workers NUM_WORKERS
                        Dataloaders threads (default: 12)
  --batch-size BATCH_SIZE
                        Batch size (default: 16)
  --cuda                Use cuda
  --save                Save model to disk
  --seed SEED           Set random seed for reproducibility
  --clip_max_norm CLIP_MAX_NORM
                        gradient clipping max norm (default: 1.0
  --checkpoint CHECKPOINT
                        Path to a previous checkpoint for sequential training.
  --logs LOGS           Path to store training logs for TensorBoard.
  --num_filters NUM_FILTERS
                        Number of filters in first convolutional layer (default: 32)
  --lambda LMBDA        Rate-distortion trade-off parameter (default: 0.01)
  --fl_alpha FL_ALPHA   Class balancing weight for Focal Loss (default: 0.7)
  --fl_gamma FL_GAMMA   Focusing weight for Focal Loss (default: 2.0)
  --omega OMEGA         Geometry-color distortion tradeoff parameter (default: 0.5)
```

Usage examples:
```
python train.py -v -d "../../dataset/training_data_geo/" --val_data "../../dataset/validation_data_geo/" --model_dir "../models/Geo/Codec/0.00025/" --logs "../logs/Geo/Codec/0.00025/" --lambda 0.00025 --epochs 500 --batch-size 32 --cuda --save
```
```
python train.py -v -d "../../dataset/training_data_geo/" --val_data "../../dataset/validation_data_geo/" --model_dir "../models/Geo/Codec/0.0005/" --logs "../logs/Geo/Codec/0.0005/" --lambda 0.0005 --epochs 500 --batch-size 32 --cuda --save --checkpoint "../models/Geo/Codec/0.00025/checkpoint_best_loss.pth.tar"
```
```
python train.py -v --with_color -d "../../dataset/training_data_joint/" --val_data "../../dataset/validation_data_joint/" --model_dir "../models/Joint/Codec/0.00025/" --logs "../logs/Joint/Codec/0.00025/" --lambda 0.00025 --omega 0.5 --epochs 500 --batch-size 16 --cuda --save
```

## Training a DL super-resolution model:
```
usage: train_sr.py [-h] [-v] [--with_color] -d TRAIN_DATA --val_data VAL_DATA --model_dir MODEL_DIR [-e EPOCHS] [-n NUM_WORKERS] [--batch-size BATCH_SIZE] [--cuda] [--save] [--seed SEED]
                    [--clip_max_norm CLIP_MAX_NORM] [--logs LOGS] [--num_filters NUM_FILTERS] [--fl_alpha FL_ALPHA] [--fl_gamma FL_GAMMA] [--omega OMEGA] [--blk_size BLK_SIZE] [--sfactor SFACTOR]

Codec training script.

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         Print model summary.
  --with_color          Jointly compress both geometry and color.
  -d TRAIN_DATA, --train_data TRAIN_DATA
                        Path to training PC data (folder with pickle .pkl files).
  --val_data VAL_DATA   Path to validation PC data (folder with pickle .pkl files).
  --model_dir MODEL_DIR
                        Directory where to save the model checkpoints.
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs (default: 500)
  -n NUM_WORKERS, --num-workers NUM_WORKERS
                        Dataloaders threads (default: 12)
  --batch-size BATCH_SIZE
                        Batch size (default: 16)
  --cuda                Use cuda
  --save                Save model to disk
  --seed SEED           Set random seed for reproducibility
  --clip_max_norm CLIP_MAX_NORM
                        gradient clipping max norm (default: 1.0
  --logs LOGS           Path to store training logs for TensorBoard.
  --num_filters NUM_FILTERS
                        Number of filters in first convolutional layer (default: 16)
  --fl_alpha FL_ALPHA   Class balancing weight for Focal Loss (default: 0.7)
  --fl_gamma FL_GAMMA   Focusing weight for Focal Loss (default: 2.0)
  --omega OMEGA         Geometry-color distortion tradeoff parameter (default: 0.5)
  --blk_size BLK_SIZE   Training data block size (default: 64)
  --sfactor SFACTOR     Up-sampling factor (default: 2)
```

Usage examples:
```
python train_sr.py -v -d "../../dataset/training_data_geo/" --val_data "../../dataset/validation_data_geo/" --model_dir "../models/Geo/SR/SF_2/" --logs "../logs/Geo/SR/SF_2/" --sfactor 2 --blk_size 64 --epochs 500 --batch-size 16 --cuda --save
```
```
python train_sr.py -v --with_color -d "../../dataset/training_data_joint_128/" --val_data "../../dataset/validation_data_joint_128/" --model_dir "../models/Joint/SR/SF_4/" --logs "../logs/Joint/SR/SF_4/" --sfactor 4 --blk_size 128 --omega 0.5 --epochs 500 --batch-size 2 --cuda --save
```



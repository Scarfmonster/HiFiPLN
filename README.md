# HiFiPLN
**Multispeaker Community Vocoder model for [DiffSinger](https://github.com/openvpi/DiffSinger)**

This is the code used to train the "HiFiPLN" vocoder.

A trained model for use with OpenUtau is available for download on the official [release page](https://utau.pl/hifipln/).

## Why HiFiPLN?
Because a lot of PLN was spent training this thing.

## Training
### Python
Python 3.10 or 3.11 is required.

### Data preparation
Preperocessing and splitting the dataset into smaller files is done using a single script. Note that if the input files are shorter than `--length` seconds, they will be skipped. It is better to provide full unsegmented files to the script, but if your input files are already split into chunks, you can run with `--length 0` to disable splitting.
```bash
python preproc.py --config PATH_TO_CONFIG -o "dataset/train" --length 1 PATH_TO_TRAIN_DATASET
```
You will also need to provide some validation audio files. Run `preproc.py` with `--length 0` to disable segmenting.
```bash
python preproc.py --config PATH_TO_CONFIG -o "dataset/valid" --length 0 PATH_TO_VALIDATION_DATASET
```

### Train model
```bash
python train.py --config "configs/hifipln.yaml"
```
* If you see an error saying "Total length of \`Data Loader\` across ranks is zero" then you do not have enough validation files.
* You may want to edit `configs/hifipln.yaml` and change `train: batch_size: 12` to a value that better fits your available VRAM.

### Resume 
```bash
python train.py --config "configs/hifipln.yaml" --resume CKPT_PATH
```
You may set CKPT_PATH to a log directory (eg. logs/HiFiPLN), and it will find the last checkpoint of the last run.

### Finetuning
Download a checkpoint from https://utau.pl/hifipln/#checkpoints-for-finetuning \
Save the checkpoint as ckpt/HiFiPLN.ckpt then run:
```bash
python train.py --config "configs/hifipln-finetune.yaml"
```
* Finetuning shouldn't be run for too long, especially for small datasets. Just 2-3 epochs or ~20000 steps should be fine.

## Exporting for use in OpenUtau
```bash
python export.py --config configs/hifipln.yaml --output out/hifipln --model CKPT_PATH
```
You may set CKPT_PATH to a log directory (eg. logs/HiFiPLN), and it will find the last checkpoint of the last run.

# Credits
* [DiffSinger](https://github.com/openvpi/DiffSinger)
* [Fish Diffusion](https://github.com/fishaudio/fish-diffusion)
* [HiFi-GAN](https://github.com/jik876/hifi-gan) ([Paper](https://arxiv.org/abs/2010.05646))
* [OpenVPI](https://github.com/openvpi/SingingVocoders)
* [PC-DDSP](https://github.com/yxlllc/pc-ddsp)
* [RefineGAN](https://arxiv.org/abs/2111.00962)
* [UnivNet](https://github.com/maum-ai/univnet) ([Paper](https://arxiv.org/abs/2106.07889))
* [FA-GAN](https://arxiv.org/abs/2407.04575)
* [APNet2](https://arxiv.org/abs/2311.11545)

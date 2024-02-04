# HiFiPLN
**Multispeaker Community Vocoder model for [DiffSinger](https://github.com/openvpi/DiffSinger)**

This is the code used to train the "HiFiPLN" vocoder.

A trained model for use with OpenUtau is available for download on the official [release page](https://utau.pl/hifipln/).

## Why HiFiPLN?
Because a lot of PLN was spent training this thing.

## Training
### Python
Python 3.10 or greater is required.

### Data preparation
```bash
python dataset-utils/split.py --length 1 -sr 44100 -o "dataset/train" PATH_TO_DATASET
```
You will also need to provide some validation audio files and save them to `dataset/valid` and then run:
```bash
python preproc.py --path dataset --config "configs/hifipln.yaml"
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
Save the base checkpoint as ckpt/HiFiPLN.ckpt then run:
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

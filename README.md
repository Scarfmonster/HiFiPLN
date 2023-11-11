# HiFiPLN
**Multispeaker Community Vocoder model for [DiffSinger](https://github.com/openvpi/DiffSinger)**

This is the code used to train the "HiFiPLN" vocoder.

Pretrained model is available to download on the official [release page](https://utau.pl/hifipln/).

## Training
### Data preparation
```bash
python preproc.py --path dataset --clean --config "configs/hifipln.yaml"
```
Please note that dataset prepared using the `hifipln.yaml` config is not compatible for VUV model training. To train the VUV model either prepare the dataset using it's config, or set `vuv: True` in `hifipln.yaml`.

### VUV & Power models
To train the VUV and Power submodels run
```bash
python train.py --config "configs/vuv.yaml"
python train.py --config "configs/power.yaml"
```
Set the paths to the back checkpoints in `hifipln.yaml`.

### Main model
```bash
python train.py --config "configs/hifipln.yaml"
```

## Exporting for use in OpenUtau
```bash
python export.py --config configs/hifipln.yaml --output out/hifipln --model CKPT_PATH
```

# Credits
* [DiffSinger](https://github.com/openvpi/DiffSinger)
* [HiFi-GAN](https://github.com/jik876/hifi-gan) ([Paper](https://arxiv.org/abs/2010.05646))
* [UnivNet](https://github.com/maum-ai/univnet) ([Paper](https://arxiv.org/abs/2106.07889))
* [RefineGAN](https://arxiv.org/abs/2111.00962)
* [Fish Diffusion](https://github.com/fishaudio/fish-diffusion)

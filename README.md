# HiFiPLN
**Multispeaker Community Vocoder model for [DiffSinger](https://github.com/openvpi/DiffSinger)**

This is the code used to train the "HiFiPLN" vocoder.

Pretrained model is available for download on the official [release page](https://utau.pl/hifipln/).

## Why HiFiPLN?
Because a lot of PLN was spent training this thing.

## Training
### Data preparation
```bash
python preproc.py --path dataset --clean --config "configs/hifipln.yaml"
```

### Main model
```bash
python train.py --config "configs/hifipln.yaml"
```

### Finetuning
Save the base checkpoint as ckpt/HiFiPLN.cktp then run:
```bash
python train.py --config "configs/hifipln-finetune.yaml"
```

## Exporting for use in OpenUtau
```bash
python export.py --config configs/hifipln.yaml --output out/hifipln --model CKPT_PATH
```

# Credits
* [DiffSinger](https://github.com/openvpi/DiffSinger)
* [Fish Diffusion](https://github.com/fishaudio/fish-diffusion)
* [HiFi-GAN](https://github.com/jik876/hifi-gan) ([Paper](https://arxiv.org/abs/2010.05646))
* [OpenVPI](https://github.com/openvpi/SingingVocoders)
* [PC-DDSP](https://github.com/yxlllc/pc-ddsp)
* [RefineGAN](https://arxiv.org/abs/2111.00962)
* [UnivNet](https://github.com/maum-ai/univnet) ([Paper](https://arxiv.org/abs/2106.07889))

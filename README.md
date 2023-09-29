# Models

We use the following models from [Open CLIP](https://github.com/mlfoundations/open_clip):
* ViT CLIP:
  * modelname: ViT-B-16
  * pretrained: laion400m_e31
* ConvNeXt CLIP:
  * modelname: convnext_base
  * pretrained: laion400m_s13b_b51k

For supervised models we use the implementation from the original repo:
* [DeiT3-Base/16 Repo](https://github.com/facebookresearch/deit/blob/main/README_revenge.md)
  * weights file link: [download](https://dl.fbaipublicfiles.com/deit/deit_3_base_224_21k.pth)
* [ConvNeXt V1 Base Repo](https://github.com/facebookresearch/ConvNeXt)
  * weights file link: [download](https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth)

# Installation

You can install the needed packages with `pip install requirements.txt` although some additional tinkering may be required.

## Datasets

* [ImageNet-X](https://facebookresearch.github.io/imagenetx/site/home)
* [PuG-ImageNet](https://github.com/facebookresearch/PUG/tree/main/PUG_ImageNet)
* [EasyRobust](https://github.com/alibaba/easyrobust) for robustness datasets
* [ImageNet-Hard](https://huggingface.co/datasets/taesiri/imagenet-hard)
* [ImageNet-Real annotations](https://github.com/google-research/reassessed-imagenet/blob/master/real.json)

# ImageNet-X

To run `ImageNet-X` eval you can use the following command:

```
python3 main.py \
    --model "deit3_21k" \
    --experiment "imagenetx" \
    --data_path $data_path
```
# PuG-ImageNet

To run `PuG-ImageNet` eval you can use the following command:

```
python3 main.py \
    --model "deit3_21k" \
    --experiment "pug_imagenet" \
    --data_path $data_path
```

# Robustness

`ImageNet-Hard` and `ImageNet-Real` need to be run separately.

ImageNet-Hard can be run as:
```
python3 main.py \
    --model "deit3_21k" \
    --experiment "imagenethard" \
    --data_path $data_path
```

ImageNet-Real can be run as:
```
python3 main.py \
    --model "deit3_21k" \
    --experiment "imagenet_real" \
    --data_path $data_path
```

All other benchmarks can be run with a single command using `easyrobust` library:
```
python3 main.py \
    --model "deit3_21k" \
    --experiment "robustness" \
    --data_path $data_path
```

# Transformation Invariance

Scale invariance:
```
python3 main.py \
    --model "deit3_21k" \
    --experiment "scale" \
    --data_path $data_path
```

Shift invariance:
```
python3 main.py \
    --model "deit3_21k" \
    --experiment "shift" \
    --data_path $data_path
```

Resolution invariance:
```
python3 main.py \
    --model "deit3_21k" \
    --experiment "resolution" \
    --data_path $data_path
```

# Acknowledgments

Code for other experiments will be open-sourced later. For them we use the following codebases:

* Transferability: we follow linear probing protocol from [CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark)
* Shape-texture bias: we use [model-vs-human](https://github.com/bethgelab/model-vs-human) library and its fork from [EasyRobust](https://github.com/alibaba/easyrobust)
* To compute CKA we use implementation from [Pytorch-Model-Compare](https://github.com/AntixK/PyTorch-Model-Compare)
* Calibration: we use [calibration-library](https://github.com/Jonathan-Pearce/calibration_library)

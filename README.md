# Fork of [GuoLanqing/ShadowFormer](https://github.com/GuoLanqing/ShadowFormer)

Differences between original repository and fork:

* Compatibility with PyTorch >=2.5. (🔥)
* Original pretrained models and converted ONNX models from GitHub [releases page](https://github.com/clibdev/ShadowFormer/releases). (🔥)
* Model conversion to ONNX format using the [export.py](export.py) file. (🔥)
* Sample script [inference.py](inference.py) for inference of single image.
* The following deprecations has been fixed:
  * UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.
  * FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers.
  * FutureWarning: You are using 'torch.load' with 'weights_only=False'.

# Installation

```shell
pip install -r requirements.txt
```

# Pretrained models

* Download links:

| Name                 | Model Size (MB) | Link                                                                                                                                                                                                          | SHA-256                                                                                                                              |
|----------------------|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| ShadowFormer (ISTD)  | 130.9<br>83.0   | [PyTorch](https://github.com/clibdev/ShadowFormer/releases/latest/download/shadowformer-istd.pt)<br>[ONNX](https://github.com/clibdev/ShadowFormer/releases/latest/download/shadowformer-istd.onnx)           | 4700ae374b965253734dbcac0b63c9cac9af5895ff19655710042a988751fc98<br>96b90f5f1d11b67e3c7835cae3ccacaaa78ac4fadbf03a04fd36769e21f619a6 |
| ShadowFormer (ISTD+) | 130.9<br>83.0   | [PyTorch](https://github.com/clibdev/ShadowFormer/releases/latest/download/shadowformer-istd-plus.pt)<br>[ONNX](https://github.com/clibdev/ShadowFormer/releases/latest/download/shadowformer-istd-plus.onnx) | 2748060149908df37cc65f0695ef61d64cd25847aba0c35af36823f9b780f5b2<br>077128017e7400c0e7c22210d6afb83748bfb068a6e02037156ea4ab8a8592a9 |

# Inference

```shell
python inference.py --weights shadowformer-istd.pt --input_path img/noisy_image.png --mask_path img/mask.png
python inference.py --weights shadowformer-istd-plus.pt --input_path img/noisy_image.png --mask_path img/mask.png
```

# Export to ONNX format

```shell
pip install onnx
```
```shell
python export.py --weights shadowformer-istd.pt
python export.py --weights shadowformer-istd-plus.pt
```

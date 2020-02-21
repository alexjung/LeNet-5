# ParalleNet

This project implements several network models:
  * A modified LeNet-5 [1] with parallel layers and achieves an accuracy of ~99% on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
  * A modified version of this net using an Inception-Layer [2] achieving an accuracy of ~99% on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/)


![Epoch Train Loss visualization](https://i.imgur.com/h4h7CrF.gif)

## Setup

Install all dependencies using the following command

```
$ pip install -r requirements.txt
```

## Usage

[Optional] Start the `visdom` server for visualization

```
$ python -m visdom.server
```

Start the training procedure

```
$ python run.py MODEL DEVICE
```
with `MODEL=parallenet/mininception` and `DEVICE=cpu/cuda`

See epoch train loss live graph at [`http://localhost:8097`](http://localhost:8097).

The trained model will be exported as ONNX to `lenet.onnx`. The `lenet.onnx` file can be viewed with [Netron](https://www.electronjs.org/apps/netron).

## References

[[1](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998.  
[[2](https://arxiv.org/abs/1409.4842)] Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
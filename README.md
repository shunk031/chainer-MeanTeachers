# chainer-MeanTeachers

This repository contains a [Chainer](https://chainer.org/) implementation for the paper: [Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results](https://arxiv.org/abs/1703.01780) (NIPS 2017, Tarvainen, Antti, and Harri Valpola).

![Figure 2](https://raw.githubusercontent.com/shunk031/chainer-MeanTeachers/master/img/figure2.png)

## Requirements

- Chainer 4.0+
- CuPy 4.0+

## Notes when using

- Slightly modify your model architecture like [this](https://github.com/shunk031/chainer-MeanTeachers/blob/master/models/resnet.py#L68)

## Reference

- [Tarvainen, Antti, and Harri Valpola. "Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results." arXiv preprint arXiv:1703.01780 (2017).](https://arxiv.org/abs/1703.01780)

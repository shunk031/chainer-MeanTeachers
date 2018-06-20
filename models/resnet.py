import collections
import sys
import os

import chainer
import chainer.functions as F
import chainer.links as L

from chainer import initializers


class ResNetLayers(chainer.Chain):

    def __init__(self, pretrained_model, n_layers, downsample_fb=False):
        super(ResNetLayers, self).__init__()

        if pretrained_model:
            # As a sampling process is time-consuming,
            # we employ a zero initializer for faster computation.
            conv_kwargs = {'initialW': initializers.constant.Zero()}
        else:
            # employ default initializers used in the original paper
            conv_kwargs = {'initialW': initializers.normal.HeNormal(scale=1.0)}

        kwargs = conv_kwargs.copy()
        kwargs['downsample_fb'] = downsample_fb

        if n_layers == 50:
            block = [3, 4, 6, 3]
        elif n_layers == 101:
            block = [3, 4, 23, 3]
        elif n_layers == 152:
            block = [3, 8, 36, 3]
        else:
            raise ValueError('The n_layers argument should be either 50, 101,'
                             ' or 152, but {} was given.'.format(n_layers))

        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 64, 7, 2, 3, **conv_kwargs)
            self.bn1 = L.BatchNormalization(64)
            self.res2 = BuildingBlock(block[0], 64, 64, 256, 1, **kwargs)
            self.res3 = BuildingBlock(block[1], 256, 128, 512, 2, **kwargs)
            self.res4 = BuildingBlock(block[2], 512, 256, 1024, 2, **kwargs)
            self.res5 = BuildingBlock(block[3], 1024, 512, 2048, 2, **kwargs)
            self.fc6_1 = L.Linear(2048, 1000)
            self.fc6_2 = L.Linear(2048, 1000)

        if pretrained_model and pretrained_model.endswith('.caffemodel'):
            _retrieve(n_layers, 'ResNet-{}-model.npz'.format(n_layers),
                      pretrained_model, self)
        elif pretrained_model:
            chainer.serializers.load_npz(pretrained_model, self)

    def __call__(self, x):
        h = self.bn1(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = F.average_pooling_2d(h, 7, stride=1)

        return self.fc6_1(h), self.fc6_2(h)


class ResNet50Layers(ResNetLayers):

    """A pre-trained CNN model with 50 layers provided by MSRA.
    When you specify the path of the pre-trained chainer model serialized as
    a ``.npz`` file in the constructor, this chain model automatically
    initializes all the parameters with it.
    This model would be useful when you want to extract a semantic feature
    vector per image, or fine-tune the model on a different dataset.
    Note that unlike ``VGG16Layers``, it does not automatically download a
    pre-trained caffemodel. This caffemodel can be downloaded at
    `GitHub <https://github.com/KaimingHe/deep-residual-networks>`_.
    If you want to manually convert the pre-trained caffemodel to a chainer
    model that can be specified in the constructor,
    please use ``convert_caffemodel_to_npz`` classmethod instead.
    ResNet50 has 25,557,096 trainable parameters, and it's 58% and 43% fewer
    than ResNet101 and ResNet152, respectively. On the other hand, the top-5
    classification accuracy on ImageNet dataset drops only 0.7% and 1.1% from
    ResNet101 and ResNet152, respectively. Therefore, ResNet50 may have the
    best balance between the accuracy and the model size. It would be basically
    just enough for many cases, but some advanced models for object detection
    or semantic segmentation use deeper ones as their building blocks, so these
    deeper ResNets are here for making reproduction work easier.
    See: K. He et. al., `Deep Residual Learning for Image Recognition
    <https://arxiv.org/abs/1512.03385>`_
    Args:
        pretrained_model (str): the destination of the pre-trained
            chainer model serialized as a ``.npz`` file.
            If this argument is specified as ``auto``,
            it automatically loads and converts the caffemodel from
            ``$CHAINER_DATASET_ROOT/pfnet/chainer/models/ResNet-50-model.caffemodel``,
            where ``$CHAINER_DATASET_ROOT`` is set as
            ``$HOME/.chainer/dataset`` unless you specify another value
            by modifying the environment variable. Note that in this case the
            converted chainer model is stored on the same directory and
            automatically used from the next time.
            If this argument is specified as ``None``, all the parameters
            are not initialized by the pre-trained model, but the default
            initializer used in the original paper, i.e.,
            ``chainer.initializers.HeNormal(scale=1.0)``.
        downsample_fb (bool): If this argument is specified as ``False``,
            it performs downsampling by placing stride 2
            on the 1x1 convolutional layers (the original MSRA ResNet).
            If this argument is specified as ``True``, it performs downsampling
            by placing stride 2 on the 3x3 convolutional layers
            (Facebook ResNet).
    Attributes:
        ~ResNet50Layers.available_layers (list of str): The list of available
            layer names used by ``__call__`` and ``extract`` methods.
    """

    def __init__(self, pretrained_model='auto', downsample_fb=False):
        if pretrained_model == 'auto':
            pretrained_model = 'ResNet-50-model.caffemodel'
        super(ResNet50Layers, self).__init__(
            pretrained_model, 50, downsample_fb)


class ResNet101Layers(ResNetLayers):

    """A pre-trained CNN model with 101 layers provided by MSRA.
    When you specify the path of the pre-trained chainer model serialized as
    a ``.npz`` file in the constructor, this chain model automatically
    initializes all the parameters with it.
    This model would be useful when you want to extract a semantic feature
    vector per image, or fine-tune the model on a different dataset.
    Note that unlike ``VGG16Layers``, it does not automatically download a
    pre-trained caffemodel. This caffemodel can be downloaded at
    `GitHub <https://github.com/KaimingHe/deep-residual-networks>`_.
    If you want to manually convert the pre-trained caffemodel to a chainer
    model that can be specified in the constructor,
    please use ``convert_caffemodel_to_npz`` classmethod instead.
    ResNet101 has 44,549,224 trainable parameters, and it's 43% fewer than
    ResNet152 model, while the top-5 classification accuracy on ImageNet
    dataset drops 1.1% from ResNet152. For many cases, ResNet50 may have the
    best balance between the accuracy and the model size.
    See: K. He et. al., `Deep Residual Learning for Image Recognition
    <https://arxiv.org/abs/1512.03385>`_
    Args:
        pretrained_model (str): the destination of the pre-trained
            chainer model serialized as a ``.npz`` file.
            If this argument is specified as ``auto``,
            it automatically loads and converts the caffemodel from
            ``$CHAINER_DATASET_ROOT/pfnet/chainer/models/ResNet-101-model.caffemodel``,
            where ``$CHAINER_DATASET_ROOT`` is set as
            ``$HOME/.chainer/dataset`` unless you specify another value
            by modifying the environment variable. Note that in this case the
            converted chainer model is stored on the same directory and
            automatically used from the next time.
            If this argument is specified as ``None``, all the parameters
            are not initialized by the pre-trained model, but the default
            initializer used in the original paper, i.e.,
            ``chainer.initializers.HeNormal(scale=1.0)``.
        downsample_fb (bool): If this argument is specified as ``False``,
            it performs downsampling by placing stride 2
            on the 1x1 convolutional layers (the original MSRA ResNet).
            If this argument is specified as ``True``, it performs downsampling
            by placing stride 2 on the 3x3 convolutional layers
            (Facebook ResNet).
    Attributes:
        ~ResNet101Layers.available_layers (list of str): The list of available
            layer names used by ``__call__`` and ``extract`` methods.
    """

    def __init__(self, pretrained_model='auto', downsample_fb=False):
        if pretrained_model == 'auto':
            pretrained_model = 'ResNet-101-model.caffemodel'
        super(ResNet101Layers, self).__init__(
            pretrained_model, 101, downsample_fb)


class ResNet152Layers(ResNetLayers):

    """A pre-trained CNN model with 152 layers provided by MSRA.
    When you specify the path of the pre-trained chainer model serialized as
    a ``.npz`` file in the constructor, this chain model automatically
    initializes all the parameters with it.
    This model would be useful when you want to extract a semantic feature
    vector per image, or fine-tune the model on a different dataset.
    Note that unlike ``VGG16Layers``, it does not automatically download a
    pre-trained caffemodel. This caffemodel can be downloaded at
    `GitHub <https://github.com/KaimingHe/deep-residual-networks>`_.
    If you want to manually convert the pre-trained caffemodel to a chainer
    model that can be specified in the constructor,
    please use ``convert_caffemodel_to_npz`` classmethod instead.
    ResNet152 has 60,192,872 trainable parameters, and it's the deepest ResNet
    model and it achieves the best result on ImageNet classification task in
    `ILSVRC 2015 <http://image-net.org/challenges/LSVRC/2015/results#loc>`_.
    See: K. He et. al., `Deep Residual Learning for Image Recognition
    <https://arxiv.org/abs/1512.03385>`_
    Args:
        pretrained_model (str): the destination of the pre-trained
            chainer model serialized as a ``.npz`` file.
            If this argument is specified as ``auto``,
            it automatically loads and converts the caffemodel from
            ``$CHAINER_DATASET_ROOT/pfnet/chainer/models/ResNet-152-model.caffemodel``,
            where ``$CHAINER_DATASET_ROOT`` is set as
            ``$HOME/.chainer/dataset`` unless you specify another value
            by modifying the environment variable. Note that in this case the
            converted chainer model is stored on the same directory and
            automatically used from the next time.
            If this argument is specified as ``None``, all the parameters
            are not initialized by the pre-trained model, but the default
            initializer used in the original paper, i.e.,
            ``chainer.initializers.HeNormal(scale=1.0)``.
        downsample_fb (bool): If this argument is specified as ``False``,
            it performs downsampling by placing stride 2
            on the 1x1 convolutional layers (the original MSRA ResNet).
            If this argument is specified as ``True``, it performs downsampling
            by placing stride 2 on the 3x3 convolutional layers
            (Facebook ResNet).
    Attributes:
        ~ResNet152Layers.available_layers (list of str): The list of available
            layer names used by ``__call__`` and ``extract`` methods.
    """

    def __init__(self, pretrained_model='auto', downsample_fb=False):
        if pretrained_model == 'auto':
            pretrained_model = 'ResNet-152-model.caffemodel'
        super(ResNet152Layers, self).__init__(
            pretrained_model, 152, downsample_fb)


class BuildingBlock(chainer.Chain):

    """A building block that consists of several Bottleneck layers.
    Args:
        n_layer (int): Number of layers used in the building block.
        in_channels (int): Number of channels of input arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        out_channels (int): Number of channels of output arrays.
        stride (int or tuple of ints): Stride of filter application.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
        downsample_fb (bool): If this argument is specified as ``False``,
            it performs downsampling by placing stride 2
            on the 1x1 convolutional layers (the original MSRA ResNet).
            If this argument is specified as ``True``, it performs downsampling
            by placing stride 2 on the 3x3 convolutional layers
            (Facebook ResNet).
    """

    def __init__(self, n_layer, in_channels, mid_channels,
                 out_channels, stride, initialW=None, downsample_fb=False):
        super(BuildingBlock, self).__init__()
        with self.init_scope():
            self.a = BottleneckA(
                in_channels, mid_channels, out_channels, stride,
                initialW, downsample_fb)
            self._forward = ["a"]
            for i in range(n_layer - 1):
                name = 'b{}'.format(i + 1)
                bottleneck = BottleneckB(out_channels, mid_channels, initialW)
                setattr(self, name, bottleneck)
                self._forward.append(name)

    def __call__(self, x):
        for name in self._forward:
            l = getattr(self, name)
            x = l(x)
        return x

    @property
    def forward(self):
        return [getattr(self, name) for name in self._forward]


class BottleneckA(chainer.Chain):

    """A bottleneck layer that reduces the resolution of the feature map.
    Args:
        in_channels (int): Number of channels of input arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        out_channels (int): Number of channels of output arrays.
        stride (int or tuple of ints): Stride of filter application.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
        downsample_fb (bool): If this argument is specified as ``False``,
            it performs downsampling by placing stride 2
            on the 1x1 convolutional layers (the original MSRA ResNet).
            If this argument is specified as ``True``, it performs downsampling
            by placing stride 2 on the 3x3 convolutional layers
            (Facebook ResNet).
    """

    def __init__(self, in_channels, mid_channels, out_channels,
                 stride=2, initialW=None, downsample_fb=False):
        super(BottleneckA, self).__init__()
        # In the original MSRA ResNet, stride=2 is on 1x1 convolution.
        # In Facebook ResNet, stride=2 is on 3x3 convolution.

        stride_1x1, stride_3x3 = (stride, 1) if downsample_fb else (1, stride)
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels, mid_channels, 1, stride_1x1, 0, initialW=initialW,
                nobias=True)
            self.bn1 = L.BatchNormalization(mid_channels)
            self.conv2 = L.Convolution2D(
                mid_channels, mid_channels, 3, stride_3x3, 1,
                initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(mid_channels)
            self.conv3 = L.Convolution2D(
                mid_channels, out_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn3 = L.BatchNormalization(out_channels)
            self.conv4 = L.Convolution2D(
                in_channels, out_channels, 1, stride, 0, initialW=initialW,
                nobias=True)
            self.bn4 = L.BatchNormalization(out_channels)

    def __call__(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))
        h2 = self.bn4(self.conv4(x))
        return F.relu(h1 + h2)


class BottleneckB(chainer.Chain):

    """A bottleneck layer that maintains the resolution of the feature map.
    Args:
        in_channels (int): Number of channels of input and output arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, in_channels, mid_channels, initialW=None):
        super(BottleneckB, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels, mid_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn1 = L.BatchNormalization(mid_channels)
            self.conv2 = L.Convolution2D(
                mid_channels, mid_channels, 3, 1, 1, initialW=initialW,
                nobias=True)
            self.bn2 = L.BatchNormalization(mid_channels)
            self.conv3 = L.Convolution2D(
                mid_channels, in_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn3 = L.BatchNormalization(in_channels)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        return F.relu(h + x)


def _transfer_components(src, dst_conv, dst_bn, bname, cname):
    src_conv = getattr(src, 'res{}_branch{}'.format(bname, cname))
    src_bn = getattr(src, 'bn{}_branch{}'.format(bname, cname))
    src_scale = getattr(src, 'scale{}_branch{}'.format(bname, cname))
    dst_conv.W.data[:] = src_conv.W.data
    dst_bn.avg_mean[:] = src_bn.avg_mean
    dst_bn.avg_var[:] = src_bn.avg_var
    dst_bn.gamma.data[:] = src_scale.W.data
    dst_bn.beta.data[:] = src_scale.bias.b.data


def _transfer_bottleneckA(src, dst, name):
    _transfer_components(src, dst.conv1, dst.bn1, name, '2a')
    _transfer_components(src, dst.conv2, dst.bn2, name, '2b')
    _transfer_components(src, dst.conv3, dst.bn3, name, '2c')
    _transfer_components(src, dst.conv4, dst.bn4, name, '1')


def _transfer_bottleneckB(src, dst, name):
    _transfer_components(src, dst.conv1, dst.bn1, name, '2a')
    _transfer_components(src, dst.conv2, dst.bn2, name, '2b')
    _transfer_components(src, dst.conv3, dst.bn3, name, '2c')


def _transfer_block(src, dst, names):
    _transfer_bottleneckA(src, dst.a, names[0])
    for i, name in enumerate(names[1:]):
        dst_bottleneckB = getattr(dst, 'b{}'.format(i + 1))
        _transfer_bottleneckB(src, dst_bottleneckB, name)


def _transfer_resnet50(src, dst):
    dst.conv1.W.data[:] = src.conv1.W.data
    dst.conv1.b.data[:] = src.conv1.b.data
    dst.bn1.avg_mean[:] = src.bn_conv1.avg_mean
    dst.bn1.avg_var[:] = src.bn_conv1.avg_var
    dst.bn1.gamma.data[:] = src.scale_conv1.W.data
    dst.bn1.beta.data[:] = src.scale_conv1.bias.b.data

    _transfer_block(src, dst.res2, ['2a', '2b', '2c'])
    _transfer_block(src, dst.res3, ['3a', '3b', '3c', '3d'])
    _transfer_block(src, dst.res4, ['4a', '4b', '4c', '4d', '4e', '4f'])
    _transfer_block(src, dst.res5, ['5a', '5b', '5c'])

    dst.fc6_1.W.data[:] = src.fc1000.W.data
    dst.fc6_1.b.data[:] = src.fc1000.b.data

    dst.fc6_2.W.data[:] = src.fc1000.W.data
    dst.fc6_2.b.data[:] = src.fc1000.b.data


def _transfer_resnet101(src, dst):
    dst.conv1.W.data[:] = src.conv1.W.data
    dst.bn1.avg_mean[:] = src.bn_conv1.avg_mean
    dst.bn1.avg_var[:] = src.bn_conv1.avg_var
    dst.bn1.gamma.data[:] = src.scale_conv1.W.data
    dst.bn1.beta.data[:] = src.scale_conv1.bias.b.data

    _transfer_block(src, dst.res2, ['2a', '2b', '2c'])
    _transfer_block(src, dst.res3, ['3a', '3b1', '3b2', '3b3'])
    _transfer_block(src, dst.res4,
                    ['4a'] + ['4b{}'.format(i) for i in range(1, 23)])
    _transfer_block(src, dst.res5, ['5a', '5b', '5c'])

    dst.fc6_1.W.data[:] = src.fc1000.W.data
    dst.fc6_1.b.data[:] = src.fc1000.b.data

    dst.fc6_2.W.data[:] = src.fc1000.W.data
    dst.fc6_2.b.data[:] = src.fc1000.b.data


def _transfer_resnet152(src, dst):
    dst.conv1.W.data[:] = src.conv1.W.data
    dst.bn1.avg_mean[:] = src.bn_conv1.avg_mean
    dst.bn1.avg_var[:] = src.bn_conv1.avg_var
    dst.bn1.gamma.data[:] = src.scale_conv1.W.data
    dst.bn1.beta.data[:] = src.scale_conv1.bias.b.data

    _transfer_block(src, dst.res2, ['2a', '2b', '2c'])
    _transfer_block(src, dst.res3,
                    ['3a'] + ['3b{}'.format(i) for i in range(1, 8)])
    _transfer_block(src, dst.res4,
                    ['4a'] + ['4b{}'.format(i) for i in range(1, 36)])
    _transfer_block(src, dst.res5, ['5a', '5b', '5c'])

    dst.fc6_1.W.data[:] = src.fc1000.W.data
    dst.fc6_1.b.data[:] = src.fc1000.b.data

    dst.fc6_2.W.data[:] = src.fc1000.W.data
    dst.fc6_2.b.data[:] = src.fc1000.b.data


def _make_npz(path_npz, path_caffemodel, model, n_layers):
    sys.stderr.write(
        'Now loading caffemodel (usually it may take few minutes)\n')
    sys.stderr.flush()
    if not os.path.exists(path_caffemodel):
        raise IOError(
            'The pre-trained caffemodel does not exist. Please download it '
            'from \'https://github.com/KaimingHe/deep-residual-networks\', '
            'and place it on {}'.format(path_caffemodel))
    ResNetLayers.convert_caffemodel_to_npz(path_caffemodel, path_npz, n_layers)
    chainer.serializers.load_npz(path_npz, model)

    return model


def _retrieve(n_layers, name_npz, name_caffemodel, model):
    root = chainer.dataset.download.get_dataset_root('pfnet/chainer/models/')
    path = os.path.join(root, name_npz)
    path_caffemodel = os.path.join(root, name_caffemodel)
    return chainer.dataset.download.cache_or_load_file(
        path, lambda path: _make_npz(path, path_caffemodel, model, n_layers),
        lambda path: chainer.serializers.load_npz(path, model))

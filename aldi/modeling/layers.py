from functools import partial
from typing import Optional, Callable

import numpy as np
import tensorflow as tf

tfkl = tf.keras.layers


TRANSPOSE_FNS = {1: tfkl.Conv1DTranspose,
                 2: tfkl.Conv2DTranspose}
CONV_FNS = {1: tfkl.Conv1D,
            2: tfkl.Conv2D}
UPSAMPLE_FNS = {1: tfkl.UpSampling1D,
                2: tfkl.UpSampling2D}


class UpsampleConv(tfkl.Layer):
    def __init__(self,
                 dim: int,
                 n_filters: int,
                 filter_size: int,
                 strides: int = 1,
                 padding: str = "same",
                 **kwargs):
        super().__init__(**kwargs)
        layer_fn = CONV_FNS[dim]
        self.conv = layer_fn(n_filters, filter_size, padding=padding,
                             name=self.name + "_conv")

        if strides > 1:
            upsample_fn = UPSAMPLE_FNS[dim]
            self.upsample = upsample_fn(size=strides,
                                        name=self.name + "_upsample")

        self.strides = strides

    def call(self,
             inputs,
             training=None):
        if self.strides > 1:
            upsampled = self.upsample(inputs, training=training)
        else:
            upsampled = inputs
        return self.conv(upsampled, training=training)


class NormActConv(tfkl.Layer):
    def __init__(self,
                 dim: int,
                 n_filters: int,
                 filter_size: int,
                 mode: str,
                 strides: int = 1,
                 dilation: int = 1,
                 activation: Optional[Callable] = tf.nn.gelu,
                 normalization: Optional[Callable] = None,
                 se_factor: int = 8,
                 **kwargs):
        if mode not in ["conv", "transpose", "upconv"]:
            raise ValueError(
                "Invalid mode; valid choices are 'conv', 'transpose', "
                "'upconv'.")
        super().__init__(**kwargs)
        self.dim = dim

        if mode == "conv" or strides == 1:
            layer_fn = CONV_FNS[dim]
        elif mode == "transpose":
            layer_fn = TRANSPOSE_FNS[dim]
        else:
            layer_fn = partial(UpsampleConv, dim=dim)

        self.conv = layer_fn(n_filters,
                             filter_size,
                             strides=strides,
                             dilation_rate=dilation,
                             padding="same",
                             name=self.name + "_conv_main")
        self.activation = activation
        self.normalization = normalization() if normalization is not None else None

        self.se_factor = se_factor
        if se_factor > 0:
            pool_fn = tfkl.GlobalAveragePooling1D if dim == 1 else tfkl.GlobalAveragePooling2D
            self.squeeze = tf.keras.Sequential([pool_fn(),
                                                tfkl.Dense(n_filters // se_factor,
                                                           tf.nn.gelu)])
            self.excite = tfkl.Dense(n_filters, tf.nn.sigmoid)


    def call(self,
             inputs,
             training=None):
        if self.normalization:
            normed = self.normalization(inputs, training=training)
        else:
            normed = inputs

        if self.activation:
            acted = self.activation(normed)
        else:
            acted = normed

        conved = self.conv(acted, training=training)

        if self.se_factor > 0:
            squeeze = self.squeeze(conved)
            excite = self.excite(squeeze)
            if self.dim == 1:
                excite = excite[:, None, :]
            else:
                excite = excite[:, None, None, :]

            conved = conved * excite
        return conved


class ResidualBlock(tfkl.Layer):
    def __init__(self,
                 dim: int,
                 n_filters: int,
                 filter_size: int,
                 mode: str,
                 strides: int = 1,
                 dilation: int = 1,
                 activation: Optional[Callable] = tf.nn.gelu,
                 normalization: Optional[Callable] = None,
                 use_shortcut: bool = True,
                 rescale_residual: bool = False,
                 second_is1x1: bool = False,
                 stride_is_second: bool = False,
                 **kwargs):
        super().__init__(**kwargs)

        if isinstance(n_filters, int):
            n_filters1 = n_filters2 = n_filters
        else:
            n_filters1, n_filters2 = n_filters

        if isinstance(filter_size, int):
            filter_size1 = filter_size2 = filter_size
        else:
            filter_size1, filter_size2 = filter_size

        if stride_is_second:
            strides1, strides2 = 1, strides
        else:
            strides1, strides2 = strides, 1

        self.main_layer1 = NormActConv(dim,
                                       n_filters1,
                                       filter_size1,
                                       mode,
                                       strides=strides1,
                                       dilation=dilation,
                                       activation=activation,
                                       normalization=normalization,
                                       name=self.name + "_main1")

        self.main_layer2 = NormActConv(dim,
                                       n_filters2,
                                       1 if second_is1x1 else filter_size2,
                                       mode,
                                       strides=strides2,
                                       activation=activation,
                                       normalization=normalization,
                                       name=self.name + "_main2")

        if mode == "conv" or strides == 1:
            shortcut_fn = CONV_FNS[dim]
        elif mode == "upconv":
            shortcut_fn = partial(UpsampleConv, dim=dim)
        else:
            shortcut_fn = TRANSPOSE_FNS[dim]

        if use_shortcut:
            self.shortcut = shortcut_fn(n_filters2, 1, strides=strides,
                                        name=self.name + "_shortcut")
        self.use_shortcut = use_shortcut
        self.rescale_residual = rescale_residual

    def call(self,
             inputs,
             training=None):
        l1 = self.main_layer1(inputs, training=training)
        l2 = self.main_layer2(l1, training=training)

        if self.use_shortcut:
            shortcut = self.shortcut(inputs, training=training)
        else:
            shortcut = inputs

        # print(inputs.shape, l1.shape, l2.shape, shortcut.shape)
        # print(inputs.shape, l1.shape, l2.shape, shortcut.shape)

        out = l2 + shortcut
        if self.rescale_residual:
            out = 1./np.sqrt(2) * out

        return out


class DownLevel(tfkl.Layer):
    def __init__(self,
                 dim: int,
                 n_blocks: int,
                 block_filters: int,
                 end_filters: int,
                 filter_size: int,
                 strides: int = 1,
                 dilation_base: int = 1,
                 activation: Optional[Callable] = tf.nn.gelu,
                 normalization: Optional[Callable] = None,
                 use_shortcut: bool = True,
                 rescale_residual: bool = False,
                 second_is1x1: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.blocks = []
        for ind in range(n_blocks):
            self.blocks.append(
                ResidualBlock(dim,
                              block_filters,
                              filter_size,
                              "conv",
                              strides=1,
                              dilation=dilation_base**ind,
                              activation=activation,
                              normalization=normalization,
                              use_shortcut=use_shortcut,
                              rescale_residual=rescale_residual,
                              second_is1x1=second_is1x1,
                              name=self.name + "block{}".format(ind+1)))
        end_fn = CONV_FNS[dim]
        self.conv_end = end_fn(end_filters,
                               2*strides,
                               strides=strides,
                               padding="same",
                               name=self.name + "final_conv")

    def call(self,
             inputs,
             training=None):
        for layer in self.blocks:
            inputs = layer(inputs, training=training)
        return self.conv_end(inputs, training=training)


class UpLevel(tfkl.Layer):
    def __init__(self,
                 dim: int,
                 n_blocks: int,
                 block_filters: int,
                 start_filters: int,
                 filter_size: int,
                 strides: int = 1,
                 dilation_base: int = 1,
                 activation: Optional[Callable] = tf.nn.gelu,
                 normalization: Optional[Callable] = None,
                 use_shortcut: bool = True,
                 rescale_residual: bool = False,
                 second_is1x1: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.blocks = []
        for ind in range(n_blocks):
            # TODO allow for upsample-conv instead of transpose
            self.blocks.append(
                ResidualBlock(dim,
                              block_filters,
                              filter_size,
                              "transpose",
                              strides=1,
                              dilation=dilation_base**ind,
                              activation=activation,
                              normalization=normalization,
                              use_shortcut=use_shortcut,
                              rescale_residual=rescale_residual,
                              second_is1x1=second_is1x1,
                              name=self.name + "block{}".format(ind+1)))
        start_fn = TRANSPOSE_FNS[dim]
        self.conv_initial = start_fn(start_filters,
                                     2*strides,
                                     strides=strides,
                                     padding="same",
                                     name=self.name + "initial_conv")

    def call(self,
             inputs,
             training=None):
        inputs = self.conv_initial(inputs, training=training)
        for layer in self.blocks:
            inputs = layer(inputs, training=training)
        return inputs

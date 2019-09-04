import tensorflow as tf
from tensorflow._api.v1.layers import Layer



class ReflectionPadding2D(Layer):
    """ Assumes channels-last """
    def __init__(self, padding=(1, 1), **kwargs):
        super(ReflectionPadding2D, self).__init__()
        self.w_pad, self.h_pad = padding
        self.paddings = tf.constant([[0, 0], [self.h_pad, self.h_pad], [self.w_pad, self.w_pad], [0, 0]])

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1] + 2 * self.h_pad, input_shape[2] + 2 * self.w_pad, input_shape[3]

    def call(self, inputs, **kwargs):
        return tf.pad(inputs, paddings=self.paddings, mode='REFLECT')


class _Conv2D(Layer):
    def __init__(self, *args, **kwargs):
        super(_Conv2D, self).__init__()
        self.reflect_pad = False
        padding = kwargs.get('padding', '')
        if padding == 'zero':
            kwargs['padding'] = 'same'
        if padding == 'reflect':
            kernel_size = kwargs['kernel_size']
            if (kernel_size % 2) == 1:
                self.pad = (kernel_size // 2,) * 2
                kwargs['padding'] = 'valid'
                self.reflect_pad = True
        self.func = tf.keras.layers.Conv2D(*args, **kwargs)

    def call(self, inputs, **kwargs):
        if self.reflect_pad:
            inputs = ReflectionPadding2D(self.pad)(inputs)
        return self.func(inputs)


class _Act(Layer):
    def __init__(self, act='', lrelu_alpha=0.1, **kwargs):
        super(_Act, self).__init__(**kwargs)

        if act == 'prelu':
            self.func = tf.keras.layers.PReLU()
        else:
            self.func = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)

    def call(self, inputs, **kwargs):
        return self.func(inputs)


class _Norm(Layer):
    def __init__(self, norm='', **kwargs):
        super(_Norm, self).__init__(**kwargs)
        self.norm = norm
        # self.norm = (norm == 'bn')
        # if self.norm:
        #     self.func = tf.keras.layers.BatchNormalization(axis=-1)

    def call(self, inputs, **kwargs):
        if self.norm == 'bn':
            inputs = tf.keras.layers.BatchNormalization(axis=-1)(inputs)
        return inputs
        # if self.norm:
        #     inputs = self.func(inputs)
        # return inputs


class PixelShuffler(Layer):
    """ Assumes channels-last """
    def __init__(self, size=(2, 2), **kwargs):
        super(PixelShuffler, self).__init__(**kwargs)
        self.block_size = size[0]

    def call(self, inputs, **kwargs):
        return tf.nn.depth_to_space(inputs, self.block_size, data_format='NHWC')

    def compute_output_shape(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(f'Inputs should have rank 4; Received input shape: {input_shape}')
        height = input_shape[1] * self.size[0] if input_shape[1] is not None else None
        width = input_shape[2] * self.size[1] if input_shape[2] is not None else None
        channels = input_shape[3] // self.size[0] // self.size[1]

        if channels * self.size[0] * self.size[1] != input_shape[3]:
            raise ValueError('channels of input and size are incompatible')

        return (input_shape[0],
                height,
                width,
                channels)


class Downscale(Layer):
    def __init__(self, dim, padding='zero', norm='', act='', **kwargs):
        super(Downscale, self).__init__(**kwargs)
        self.conv_2d = _Conv2D(dim, kernel_size=5, strides=2, padding=padding)
        self.act = _Act(act)
        self.norm = _Norm(norm)

    def call(self, inputs, **kwargs):
        x = self.conv_2d(inputs)
        x = self.act(x)
        x = self.norm(x)
        return x


class Upscale(Layer):
    def __init__(self, dim, padding='zero', norm='', act='', **kwargs):
        super(Upscale, self).__init__(**kwargs)
        self.conv_2d = _Conv2D(dim * 4, kernel_size=3, strides=1, padding=padding)
        self.act = _Act(act)
        self.norm = _Norm(norm)
        self.pixel_shuffler = PixelShuffler()

    def call(self, inputs, **kwargs):
        x = self.conv_2d(inputs)
        x = self.act(inputs)
        x = self.norm(inputs)
        x = self.pixel_shuffler(x)
        return x


class ResidualBlock(Layer):
    def __init__(self, filters, kernel_size=3, padding='zero', norm='', act='', **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv_2d_1 = _Conv2D(filters, kernel_size=kernel_size, padding=padding)
        self.act_1 = _Act(act, lrelu_alpha=0.2)
        self.norm_1 = _Norm(norm)
        self.conv_2d_2 = _Conv2D(filters, kernel_size=kernel_size, padding=padding)
        self.add = tf.keras.layers.Add()
        self.act_2 = _Act(act, lrelu_alpha=0.2)
        self.norm_2 = _Norm(norm)

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.conv_2d_1(x)
        x = self.act_1(x)
        x = self.norm_1(x)
        x = self.conv_2d_2(x)
        x = self.add([x, inputs])
        x = self.act_2(x)
        x = self.norm_2(x)
        return x


class ToBgr(Layer):
    def __init__(self, num_channels, padding='zero', **kwargs):
        super(ToBgr, self).__init__(**kwargs)
        self.conv_2d = _Conv2D(num_channels, kernel_size=5, padding=padding, activation='sigmoid')

    def call(self, inputs, **kwargs):
        return self.conv_2d(inputs)


class Encoder(tf.keras.Model):
    def __init__(self,
                 num_channels=1,
                 resolution=32,
                 ch_dims=1,
                 ae_dims=1,
                 name='encoder',
                 **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        lowest_dense_res = resolution // 16
        dims = num_channels * ch_dims

        self.downscale_1 = Downscale(dims)
        self.downscale_2 = Downscale(dims * 2)
        self.downscale_3 = Downscale(dims * 4)
        self.downscale_4 = Downscale(dims * 8)
        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(ae_dims)
        self.dense_2 = tf.keras.layers.Dense(lowest_dense_res**2 * ae_dims)
        self.reshape = tf.keras.layers.Reshape((lowest_dense_res, lowest_dense_res, ae_dims))
        self.upscale = Upscale(ae_dims)

    def call(self, inputs, **kwargs):
        x = self.downscale_1(inputs)
        x = self.downscale_2(x)
        x = self.downscale_3(x)
        x = self.downscale_4(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.reshape(x)
        x = self.upscale(x)
        return x


class Decoder(tf.keras.Model):
    def __init__(self,
                 num_channels=1,
                 ch_dims=1,
                 add_residual_blocks=True,
                 multiscale_count=3,
                 name='decoder',
                 **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        dims = num_channels * ch_dims

        self.upscale_1 = Upscale(dims * 8, **kwargs)
        self.upscale_2 = Upscale(dims * 4, **kwargs)
        self.upscale_3 = Upscale(dims * 2, **kwargs)

        self.residual_blocks = []
        if add_residual_blocks:
            self.residual_blocks = [ResidualBlock(dims * 8, **kwargs),
                                    ResidualBlock(dims * 8, **kwargs),
                                    ResidualBlock(dims * 4, **kwargs),
                                    ResidualBlock(dims * 4, **kwargs),
                                    ResidualBlock(dims * 2, **kwargs),
                                    ResidualBlock(dims * 2, **kwargs)]

        self.to_bgr = [ToBgr(num_channels, **kwargs) for _ in range(multiscale_count)]

    def call(self, inputs, **kwargs):
        outputs = []
        x1 = self.upscale_1(inputs)
        if self.residual_blocks:
            x1 = self.residual_blocks[0](x1)
            x1 = self.residual_blocks[1](x1)
        if len(self.to_bgr) > 2:
            outputs += [self.to_bgr[2](x1)]

        x2 = self.upscale_2(x1)
        if self.residual_blocks:
            x2 = self.residual_blocks[2](x2)
            x2 = self.residual_blocks[3](x2)
        if len(self.to_bgr) > 1:
            outputs += [self.to_bgr[1](x2)]

        x3 = self.upscale_3(x2)
        if self.residual_blocks:
            x3 = self.residual_blocks[4](x3)
            x3 = self.residual_blocks[5](x3)
        outputs += [self.to_bgr[0](x3)]

        return outputs


class MaskDecoder(tf.keras.Model):
    def __init__(self, num_channels=1, ch_dims=1, **kwargs):
        super(MaskDecoder, self).__init__(**kwargs)
        dims = num_channels * ch_dims
        self.upscale_1 = Upscale(dims * 8)
        self.upscale_2 = Upscale(dims * 4)
        self.upscale_3 = Upscale(dims * 2)
        self.to_bgr = ToBgr(num_channels)

    def call(self, inputs, **kwargs):
        x = self.upscale_1(inputs)
        x = self.upscale_2(x)
        x = self.upscale_3(x)
        x = self.to_bgr(x)
        return x


class SparseAutoEncoder(tf.keras.Model):
    def __init__(self,
                 num_channels=3,
                 resolution=32,
                 ae_dims=16,
                 e_ch_dims=42,
                 d_ch_dims=21,
                 multiscale_count=3,
                 add_residual_blocks=True,
                 **kwargs):
        super(SparseAutoEncoder, self).__init__(**kwargs)
        self.encoder = Encoder(num_channels=num_channels, resolution=resolution, ch_dims=e_ch_dims, ae_dims=ae_dims)
        self.decoder = Decoder(num_channels=num_channels, ch_dims=d_ch_dims, add_residual_blocks=add_residual_blocks, multiscale_count=multiscale_count)

    def call(self, inputs, **kwargs):
        z = self.encoder(inputs)
        # print('Shape:', tf.shape(z))
        reconstructed = self.decoder(z)
        return reconstructed


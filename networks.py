import tensorflow as tf
# from tensorflow.keras.layers import Layer, Conv2D, PReLU, LeakyReLU, BatchNormalization, Add, Flatten, Dense, Reshape
# from tensorflow.keras import Model, Input
from tensorflow.python.keras.layers import Layer, Conv2D, PReLU, LeakyReLU, BatchNormalization, Add, Flatten, Dense, Reshape
from tensorflow.python.keras import Model, Input


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
        self.func = Conv2D(*args, **kwargs)

    def call(self, inputs, **kwargs):
        if self.reflect_pad:
            inputs = ReflectionPadding2D(self.pad)(inputs)
        return self.func(inputs)

    def compute_output_shape(self, input_shape):
        if self.reflect_pad:
            input_shape = ReflectionPadding2D(self.pad).compute_output_shape(input_shape)
        return self.func.compute_output_shape(input_shape)


class _Act(Layer):
    def __init__(self, act='', lrelu_alpha=0.1, **kwargs):
        super(_Act, self).__init__(**kwargs)

        if act == 'prelu':
            self.func = PReLU()
        else:
            self.func = LeakyReLU(alpha=lrelu_alpha)

    def call(self, inputs, **kwargs):
        return self.func(inputs)

    def compute_output_shape(self, input_shape):
        return self.func.compute_output_shape(input_shape)


class _Norm(Layer):
    def __init__(self, norm='', **kwargs):
        super(_Norm, self).__init__(**kwargs)
        self.norm = norm
        # self.norm = (norm == 'bn')
        # if self.norm:
        #     self.func = tf.keras.layers.BatchNormalization(axis=-1)

    def call(self, inputs, **kwargs):
        if self.norm == 'bn':
            inputs = BatchNormalization(axis=-1)(inputs)
        return inputs

    def compute_output_shape(self, input_shape):
        if self.norm == 'bn':
            input_shape = BatchNormalization(axis=-1).compute_output_shape(input_shape)
        return input_shape


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
    def __init__(self, dim, **kwargs):
        super(Downscale, self).__init__(**kwargs)
        self.conv_2d = Conv2D(dim, kernel_size=5, strides=2, padding='same')
        self.act = LeakyReLU(alpha=0.1)

    def call(self, inputs, **kwargs):
        x = self.conv_2d(inputs)
        x = self.act(x)
        return x

    def compute_output_shape(self, input_shape):
        input_shape = self.conv_2d.compute_output_shape(input_shape)
        input_shape = self.act.compute_output_shape(input_shape)
        return input_shape


class Upscale(Layer):
    def __init__(self, dim, padding='zero', norm='', act='', **kwargs):
        super(Upscale, self).__init__(**kwargs)
        self.conv_2d = _Conv2D(dim * 4, kernel_size=3, strides=1, padding=padding)
        self.act = _Act(act)
        self.norm = _Norm(norm)
        self.pixel_shuffler = PixelShuffler()

    def call(self, inputs, **kwargs):
        x = self.conv_2d(inputs)
        x = self.act(x)
        x = self.norm(x)
        x = self.pixel_shuffler(x)
        return x

    def compute_output_shape(self, input_shape):
        input_shape = self.conv_2d.compute_output_shape(input_shape)
        input_shape = self.act.compute_output_shape(input_shape)
        input_shape = self.norm.compute_output_shape(input_shape)
        input_shape = self.pixel_shuffler.compute_output_shape(input_shape)
        return input_shape


class ResidualBlock(Layer):
    def __init__(self, filters, kernel_size=3, padding='zero', norm='', act='', **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv_2d_1 = _Conv2D(filters, kernel_size=kernel_size, padding=padding)
        self.act_1 = _Act(act, lrelu_alpha=0.2)
        self.norm_1 = _Norm(norm)
        self.conv_2d_2 = _Conv2D(filters, kernel_size=kernel_size, padding=padding)
        self.add = Add()
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


def encoder(num_channels=1, resolution=32, ch_dims=1, ae_dims=1, name='encoder') -> Model:
    lowest_dense_res = resolution // 16
    dims = num_channels * ch_dims

    inputs = Input(shape=(resolution, resolution, num_channels), name='image')
    x = Downscale(dims)(inputs)
    x = Downscale(dims * 2)(x)
    x = Downscale(dims * 4)(x)
    x = Flatten()(x)
    x = Dense(ae_dims)(x)
    x = Dense(lowest_dense_res**2 * ae_dims)(x)
    x = Reshape((lowest_dense_res, lowest_dense_res, ae_dims))(x)
    outputs = Upscale(ae_dims)(x)
    return Model(inputs=inputs, outputs=outputs, name=name)


def decoder(num_channels=1, resolution=32, ch_dims=1, ae_dims=1, add_residual_blocks=True, multiscale_count=3, name='decoder') -> Model:
    lowest_dense_res = 2 * (resolution // 16)
    dims = num_channels * ch_dims

    inputs = Input(shape=(lowest_dense_res, lowest_dense_res, ae_dims))
    outputs = []

    x1 = Upscale(dims * 8)(inputs)
    if add_residual_blocks:
        x1 = ResidualBlock(dims * 8)(x1)
        x1 = ResidualBlock(dims * 8)(x1)
    if multiscale_count >= 3:
        outputs.append(ToBgr(num_channels)(x1))

    x2 = Upscale(dims * 4)(x1)
    if add_residual_blocks:
        x2 = ResidualBlock(dims * 4)(x2)
        x2 = ResidualBlock(dims * 4)(x2)
    if multiscale_count >= 2:
        outputs.append(ToBgr(num_channels)(x2))

    x3 = Upscale(dims * 2)(x2)
    if add_residual_blocks:
        x3 = ResidualBlock(dims * 2)(x3)
        x3 = ResidualBlock(dims * 2)(x3)
    outputs.append(ToBgr(num_channels)(x3))

    return Model(inputs=inputs, outputs=outputs, name=name)




class MaskDecoder(Model):
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


def sparse_auto_encoder(num_channels=3, resolution=32, ae_dims=16, e_ch_dims=42, d_ch_dims=21, multiscale_count=3, add_residual_blocks=True, name='SAE'):
    encoder_model = encoder(num_channels=num_channels, resolution=resolution, ch_dims=e_ch_dims, ae_dims=ae_dims)
    decoder_model = decoder(num_channels=num_channels, resolution=resolution, ch_dims=d_ch_dims, ae_dims=ae_dims,
                      add_residual_blocks=add_residual_blocks, multiscale_count=multiscale_count)

    inputs = Input(shape=(resolution, resolution, num_channels), name='image')
    z = encoder_model(inputs)
    outputs = decoder_model(z)

    return Model(inputs=inputs, outputs=outputs, name=name)


class _SparseAutoEncoder(object):
    def __init__(self,
                 num_channels=3,
                 resolution=32,
                 ae_dims=16,
                 e_ch_dims=42,
                 d_ch_dims=21,
                 multiscale_count=3,
                 add_residual_blocks=True,
                 **kwargs):
        super(_SparseAutoEncoder, self).__init__(**kwargs)
        self.encoder = encoder(num_channels=num_channels, resolution=resolution, ch_dims=e_ch_dims, ae_dims=ae_dims)
        self.decoder = decoder(num_channels=num_channels, resolution=resolution, ch_dims=d_ch_dims, ae_dims=ae_dims,
                               add_residual_blocks=add_residual_blocks, multiscale_count=multiscale_count)

    def call(self, inputs, **kwargs):
        z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed


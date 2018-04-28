from chainer import Chain
import chainer.functions as F
import chainer.links as L
from chainer.functions import resize_images


def upscale(img, factor):
    # TODO: Saling of images
    pass


def padding_size(k_size):
    """
    :param k_size: size of the conv kernel
    :return: padding width to have maintain same size of feature
             maps with kernel size = k_size
    """
    assert k_size % 2 == 1, "Kernel Size must be odd"
    return (k_size + 1) / 2


class SingleScaleGenerator(Chain):
    def __init__(self, input_size, fmaps, k_sizes, seq_len, lowest_scale=False):
        """
        :param input_size:
        :param fmaps:
        :param k_sizes:
        :param seq_len:
        :param lowest_scale:
        """
        self.input_size = input_size
        self.fmaps = fmaps
        self.k_sizes = k_sizes
        self.seq_len = seq_len
        self.lowest_scale = lowest_scale
        super(SingleScaleGenerator, self).__init__()

        net = []
        for i in range(len(self.k_sizes) - 1):
            net += [('conv' + str(i), L.Convolution2D(self.fmaps[i], self.fmaps[i + 1],
                                                      self.k_sizes[i], padding_size(self.k_sizes[i])))]
        with self.init_scope():
            for name, layer in net:
                setattr(self, name, layer)
        self.net = net

    def __call__(self, seq_input, *args, **kwargs):
        assert seq_input.shape[2] == seq_input.shape[3], "Images must be square"
        assert seq_input.shape[2] == self.input_size, "Unexpected input shape"

        if not self.lowest_scale:
            scaled_input = kwargs['scaled_input']

            # Concatenate scaled image from prev Generator Scale
            seq_input = F.concat((seq_input, scaled_input), 1)

        # Forward Prop
        for i in range(len(self.net)):
            seq_input = getattr(self, self.net[i][0])(seq_input)

            if i == len(self.net) - 1:
                seq_input = F.tanh(seq_input)
            else:
                seq_input = F.relu(seq_input)

        output = seq_input
        if not self.lowest_scale:
            output += scaled_input
        return output


class SingleScaleDiscriminator(Chain):
    def __init__(self, fmaps, k_sizes, fc_sizes):
        """
        :param fmaps:
        :param k_sizes:
        :param fc_sizes:
        """
        super(SingleScaleDiscriminator, self).__init__()
        self.fmaps = fmaps
        self.k_sizes = k_sizes
        self.fc_sizes = fc_sizes
        net = []
        with self.init_scope():
            for i in range(len(k_sizes) - 1):
                net += [('conv' + str(i), L.Convolution2D(self.fmaps[i], self.fmaps[i + 1],
                                                          self.k_sizes[i], padding_size(self.k_sizes[i])))]
            for j in range(len(fc_sizes)):
                net += [('fc'+str(j), L.Linear(out_size=fc_sizes[j]))]
        self.net = net

    def __call__(self, x, *args, **kwargs):
        for i in range(len(self.net)-1):
            x = getattr(self, self.net[i][0])(x)
            x = F.relu(x)

        x = getattr(self, self.net[-1][0])(x)
        x = F.sigmoid(x)

class Generator(Chain):
    def __init__(self, size, scale_layer_fmaps,
                 scale_kernel_sizes, input_seq_len,
                 output_seq_len=1):
        """

        :param size:
        :param scale_layer_fmaps:
        :param scale_kernel_sizes:
        :param input_seq_len:
        :param output_seq_len:
        """
        super(Generator, self).__init__()
        self.img_size = size
        self.scale_layer_fmaps = scale_layer_fmaps
        self.scale_kernel_sizes = scale_kernel_sizes
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        with self.init_scope():
            # TODO: fill
            pass

    def __call__(self, *args, **kwargs):
        # TODO : Fill
        pass

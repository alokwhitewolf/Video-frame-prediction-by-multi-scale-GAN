from chainer import Chain
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Variable
from chainer.functions import resize_images
import cupy as cp


def padding_size(k_size):
    """
    :param k_size: size of the conv kernel
    :return: padding width to have maintain same size of feature
             maps with kernel size = k_size
    """
    assert k_size % 2 == 1, "Kernel Size must be odd"
    return int((k_size - 1) / 2)


class SingleScaleGenerator(Chain):
    def __init__(self, fmaps, k_sizes, lowest_scale=False):
        """
        :param fmaps:
        :param k_sizes:
        :param lowest_scale:
        """

        self.fmaps = fmaps
        self.k_sizes = k_sizes
        self.lowest_scale = lowest_scale
        super(SingleScaleGenerator, self).__init__()

        net = []
        for i in range(len(self.k_sizes)):
            net += [('conv' + str(i), L.Convolution2D(self.fmaps[i + 1],
                                                      self.k_sizes[i], stride=1,
                                                      pad=padding_size(self.k_sizes[i])))]
        with self.init_scope():
            for name, layer in net:
                setattr(self, name, layer)
        self.net = net

    def __call__(self, seq_input, prev_input=None, *args, **kwargs):
        """

        :param seq_input:
        :param args:
        :param kwargs:
        :return:
        """
        if not self.lowest_scale:
            # Concatenate scaled image from prev Generator Scale
            scaled_input = resize_images(prev_input, (int(seq_input.shape[2]),int(seq_input.shape[3])))
            seq_input = F.concat((seq_input, scaled_input), 1)

        # Forward Prop
        for i in range(len(self.net) - 1):
            print(i)
            seq_input = getattr(self, self.net[i][0])(seq_input)
            seq_input = F.relu(seq_input)

        seq_input = getattr(self, self.net[-1][0])(seq_input)
        seq_input = F.tanh(seq_input)

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

        for i in range(len(k_sizes)):
            net += [('conv' + str(i), L.Convolution2D(self.fmaps[i], self.fmaps[i + 1],
                                                      self.k_sizes[i], stride=1))]
        for j in range(len(fc_sizes)):
            net += [('fc' + str(j), L.Linear(in_size=None, out_size=fc_sizes[j]))]
        with self.init_scope():
            for name, layer in net:
                setattr(self, name, layer)
        self.net = net

    def __call__(self, x, *args, **kwargs):
        for i in range(len(self.net) - 1):
            x = getattr(self, self.net[i][0])(x)
            x = F.relu(x)
        x = getattr(self, self.net[-1][0])(x)
        x = F.sigmoid(x)
        return x


class AdversarialModel(Chain):
    def __init__(self, g_fmaps, g_k_sizes,
                 d_fmaps, d_k_sizes, d_fc_sizes):
        """

        :param g_fmaps:
        :param g_k_sizes:
        :param d_fmaps:
        :param d_k_sizes:
        :param d_fc_sizes:
        """
        super(AdversarialModel, self).__init__()
        self.g_fmaps = g_fmaps
        self.g_k_sizes = g_k_sizes
        self.d_fmaps = d_fmaps
        self.d_k_sizes = d_k_sizes
        self.d_fc_sizes = d_fc_sizes

        assert len(d_fmaps) == len(d_k_sizes) == len(d_fmaps) \
               == len(d_k_sizes) == len(d_fc_sizes), "Check len of fmaps, k_sizes"

        no_of_scales = len(self.g_fmaps)
        for i in range(no_of_scales):
            setattr(self, "D" + str(i + 1),
                    SingleScaleDiscriminator(fmaps=d_fmaps[i], k_sizes=d_k_sizes[i],
                                             fc_sizes=d_fc_sizes[i]))
            setattr(self, "G" + str(i + 1),
                    SingleScaleGenerator(fmaps=g_fmaps[i],
                                         k_sizes=g_k_sizes[i],
                                         lowest_scale=(i == 0)))

    def __call__(self, x, *args, **kwargs):
        # x shape = [n, 15, 32, 32]
        history, frame = x

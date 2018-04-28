from chainer import Chain
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Variable
from chainer.functions import resize_images


def upscale(img, factor):
    # TODO: Scaling of images
    pass


def padding_size(k_size):
    """
    :param k_size: size of the conv kernel
    :return: padding width to have maintain same size of feature
             maps with kernel size = k_size
    """
    assert k_size % 2 == 1, "Kernel Size must be odd"
    return int((k_size - 1) / 2)


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
        for i in range(len(self.k_sizes)):
            net += [('conv' + str(i), L.Convolution2D(self.fmaps[i + 1],
                                                      self.k_sizes[i], stride=1,
                                                      pad=padding_size(self.k_sizes[i])))]
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
        for i in range(len(self.net)-1):
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
        with self.init_scope():
            for i in range(len(k_sizes) - 1):

                net += [('conv' + str(i), L.Convolution2D(self.fmaps[i], self.fmaps[i + 1],
                                                          self.k_sizes[i], stride=1,
                                                          pad=padding_size(self.k_sizes[i])))]
            for j in range(len(fc_sizes)):
                net += [('fc'+str(j), L.Linear(out_size=fc_sizes[j]))]
        self.net = net

    def __call__(self, x, *args, **kwargs):
        for i in range(len(self.net)-1):
            x = getattr(self, self.net[i][0])(x)
            x = F.relu(x)

        x = getattr(self, self.net[-1][0])(x)
        x = F.sigmoid(x)
        return x

class Model(Chain):
    def __init__(self, ):
        pass


if __name__ == "__main__":
    HIST_LEN = 4
    gen = SingleScaleGenerator(input_size = 4, fmaps = [3*(HIST_LEN), 128, 256, 128, 3], k_sizes =[3, 3, 3, 3]
                               , seq_len=HIST_LEN, lowest_scale=True)
    inp1 = Variable(np.random.randn(1,12,4,4).astype(np.float32))
    inp2 = Variable(np.random.randn(1,3,4,4).astype(np.float32))
    gen(inp1,scaled_input=inp2)


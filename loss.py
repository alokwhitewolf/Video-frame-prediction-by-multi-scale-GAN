import cupy as cp
import chainer.functions as F



def gdl(generated, truth):
    xp = cp.get_array_module(generated.data)
    wx = xp.array([1, -1], ndmin=4).astype("f")
    wy = xp.array([[1], [-1]], ndmin=4).astype("f")

    d_gx = F.convolution_2d(generated, wx)
    d_gy = F.convolution_2d(generated, wy)

    d_tx = F.convolution_2d(truth, wx)
    d_ty = F.convolution_2d(truth, wy)

    return F.sum(F.absolute(d_gx - d_tx)) + F.sum(F.absolute(d_gy - d_ty))


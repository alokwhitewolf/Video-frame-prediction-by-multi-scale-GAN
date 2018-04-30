import cupy as cp
import chainer.functions as F
import numpy as np


def gradient_loss(generated, truth):
    xp = cp.get_array_module(generated.data)
    n, c, h, w = generated.shape
    wx = xp.array([[[1, -1]]]*c, ndmin=4).astype(xp.float32)
    wy = xp.array([[[1], [-1]]]*c, ndmin=4).astype(xp.float32)

    d_gx = F.convolution_2d(generated, wx)
    d_gy = F.convolution_2d(generated, wy)

    d_tx = F.convolution_2d(truth, wx)
    d_ty = F.convolution_2d(truth, wy)

    return F.sum(F.absolute(d_gx - d_tx)) + F.sum(F.absolute(d_gy - d_ty))


def l2_loss(generated, truth):
    return F.sum(F.squared_difference(generated, truth))

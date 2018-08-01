'''Loss functions.

Author: Shen Guo-Hua <shen-gh@atr.jp>
'''


import numpy as np


def L2_loss(feat, feat0, mask=1.):
    d = feat - feat0
    loss = (d*d*mask).sum()
    grad = 2 * d * mask
    return loss, grad


def L1_loss(feat, feat0, mask=1.):
    d = feat - feat0
    loss = np.abs(d*mask).sum()
    grad = np.sign(d)*mask
    return loss, grad


def inner_loss(feat, feat0, mask=1.):
    loss = -(feat*feat0*mask).sum()
    grad = -feat0*mask
    return loss, grad


def gram(feat, mask=1.):
    feat = (feat * mask).reshape(feat.shape[0], -1)
    feat_gram = np.dot(feat, feat.T)
    return feat_gram


def gram_loss(feat, feat0, mask=1.):
    feat_size = feat.shape[:]
    N = feat_size[0]
    M = feat_size[1] * feat_size[2]
    feat_gram = gram(feat, mask)
    feat0_gram = gram(feat0, mask)
    feat = feat.reshape(N, M)
    loss = ((feat_gram - feat0_gram)**2).sum() / (4*(N**2)*(M**2))
    grad = np.dot((feat_gram - feat0_gram),
                  feat).reshape(feat_size) * mask / ((N**2)*(M**2))
    return loss, grad


def switch_loss_fun(loss_type):
    if loss_type == 'l2':
        return L2_loss
    elif loss_type == 'l1':
        return L1_loss
    elif loss_type == 'inner':
        return inner_loss
    elif loss_type == 'gram':
        return gram_loss
    else:
        raise ValueError('unknown loss function type!')

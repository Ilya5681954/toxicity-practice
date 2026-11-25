# -*- coding: utf-8 -*-
"""Нейроны «вручную» и проверка контрольного примера из математической модели (раздел 1.3 отчёта).

Сначала нейрон и свёрточный фильтр записаны без PyTorch - по формулам. Затем те же веса
загружаются в слои nn.Conv1d и nn.Linear, и результаты прямого и обратного прохода сравниваются.
"""

import math

import numpy as np
import torch
from torch import nn

# fmt: off
def relu(z):
    return max(0.0, z)
# fmt: on


# fmt: off
class Neuron:
    """Искусственный нейрон: взвешенная сумма входов плюс
    смещение и функция активации."""

    def __init__(self, weights, bias, activation=relu):
        self.w = np.asarray(weights, dtype=float)
        self.b = float(bias)
        self.f = activation

    def __call__(self, x):
        return self.f(float(np.dot(self.w, x)) + self.b)
# fmt: on


# fmt: off
class ConvNeuron:
    """Свёрточный фильтр - нейрон с общими весами, который
    скользит окном из k слов по тексту. weights имеет форму
    [k, d]: строка j умножается на вектор j-го слова окна."""

    def __init__(self, weights, bias, activation=relu):
        self.w = np.asarray(weights, dtype=float)
        self.k = self.w.shape[0]
        self.neuron = Neuron(self.w.ravel(), bias, activation)

    def __call__(self, seq):
        seq = np.asarray(seq, dtype=float)
        return [self.neuron(seq[i:i + self.k].ravel())
                for i in range(len(seq) - self.k + 1)]
# fmt: on


def softmax(z):
    e = [math.exp(v) for v in z]
    s = sum(e)
    return [v / s for v in e]


# данные контрольного примера: «ты просто дурак», d = 2, два фильтра ширины k = 2
# fmt: off
X = [[0.2, -0.1],      # ты
     [0.1, 0.3],       # просто
     [0.9, 0.8]]       # дурак
FILTERS = [([[0.5, -0.2], [0.7, 0.6]], -0.1), ([[-0.3, 0.4], [0.2, -0.5]], 0.05)]
W_OUT = [[-0.8, 0.3],  # класс 0 - «нетоксичный»
         [1.2, -0.4]]  # класс 1 - «токсичный»
# fmt: on
B_OUT = [0.1, -0.1]
LABEL = 1


def manual_example():
    """Прямой проход и градиенты по формулам раздела 1.3."""
    conv = [ConvNeuron(w, b)(X) for w, b in FILTERS]  # c_i^(f)
    h = [max(c) for c in conv]  # max-пулинг
    z = [sum(w * v for w, v in zip(row, h)) + b for row, b in zip(W_OUT, B_OUT)]
    p = softmax(z)
    loss = -math.log(p[LABEL])
    y = [1.0 if c == LABEL else 0.0 for c in range(2)]
    dz = [pi - yi for pi, yi in zip(p, y)]  # dL/dz = p - y
    dW_out = [[d * v for v in h] for d in dz]
    dh = [sum(dz[c] * W_OUT[c][f] for c in range(2)) for f in range(2)]
    # градиент первого фильтра идёт только через позицию максимума (и только если ReLU открыт)
    pos = int(np.argmax(conv[0]))
    dW_f1 = (
        [[dh[0] * v for v in X[pos + j]] for j in range(2)]
        if conv[0][pos] > 0
        else [[0, 0], [0, 0]]
    )
    return {
        'conv': conv,
        'h': h,
        'z': z,
        'p': p,
        'loss': loss,
        'dz': dz,
        'dW_out': dW_out,
        'dh': dh,
        'pos': pos,
        'dW_f1': dW_f1,
        'db_f1': dh[0] if conv[0][pos] > 0 else 0.0,
    }


def torch_example():
    """Тот же расчёт средствами PyTorch: прямой проход и автоматическое дифференцирование."""
    conv = nn.Conv1d(2, 2, kernel_size=2)
    out = nn.Linear(2, 2)
    with torch.no_grad():
        # в Conv1d веса имеют форму [фильтры, каналы, ширина окна]
        conv.weight.copy_(torch.tensor([np.asarray(w).T for w, _ in FILTERS], dtype=torch.float32))
        conv.bias.copy_(torch.tensor([b for _, b in FILTERS]))
        out.weight.copy_(torch.tensor(W_OUT))
        out.bias.copy_(torch.tensor(B_OUT))
    x = torch.tensor(X).T.unsqueeze(0)  # [1, d, n]
    c = torch.relu(conv(x))
    h = c.max(dim=2).values
    z = out(h)
    loss = nn.functional.cross_entropy(z, torch.tensor([LABEL]))
    loss.backward()
    return {
        'conv': c[0].tolist(),
        'h': h[0].tolist(),
        'z': z[0].tolist(),
        'p': torch.softmax(z, 1)[0].tolist(),
        'loss': loss.item(),
        'dW_out': out.weight.grad.tolist(),
        'dW_f1': conv.weight.grad[0].T.tolist(),
        'db_f1': conv.bias.grad[0].item(),
    }


def check():
    m, t = manual_example(), torch_example()
    for key in ('conv', 'h', 'z', 'p', 'loss', 'dW_out', 'dW_f1', 'db_f1'):
        a, b = np.asarray(m[key], dtype=float), np.asarray(t[key], dtype=float)
        assert np.allclose(a, b, atol=1e-5), (key, a, b)
    return m, t


if __name__ == '__main__':
    m, t = check()
    print('свёртка, фильтр 1:', [round(v, 4) for v in m['conv'][0]])
    print('свёртка, фильтр 2:', [round(v, 4) for v in m['conv'][1]])
    print('после max-пулинга h =', [round(v, 4) for v in m['h']])
    print('логиты z =', [round(v, 4) for v in m['z']])
    print('вероятности p =', [round(v, 4) for v in m['p']])
    print('потери L =', round(m['loss'], 4))
    print('dL/dW_out =', [[round(v, 4) for v in r] for r in m['dW_out']])
    print(
        'dL/dW (фильтр 1) =',
        [[round(v, 4) for v in r] for r in m['dW_f1']],
        'dL/db =',
        round(m['db_f1'], 4),
    )
    print('ручной расчёт совпадает с PyTorch')

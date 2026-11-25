# -*- coding: utf-8 -*-
"""Нейронные сети для классификации комментариев на токсичные и нетоксичные."""

import torch
from torch import nn

ACTIVATIONS = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'gelu': nn.GELU,
    'leaky_relu': lambda: nn.LeakyReLU(0.01),
}


# fmt: off
class TokenEmbedding(nn.Module):
    """Слой векторного представления токенов.

    mode='word'   - вектор слова из таблицы E (первый вариант);
    mode='hybrid' - вектор слова плюс среднее векторов n-грамм;
    mode='ngram'  - только символьные n-граммы.
    """

    def __init__(self, vocab_size, dim, mode='word',
                 buckets=0):
        super().__init__()
        self.mode = mode
        self.word = None
        self.gram = None
        if mode in ('word', 'hybrid'):
            self.word = nn.Embedding(vocab_size, dim,
                                     padding_idx=0)
        if mode in ('hybrid', 'ngram'):
            self.gram = nn.Embedding(buckets, dim,
                                     padding_idx=0)

    def forward(self, x, g=None):
        out = 0
        if self.word is not None:
            out = self.word(x)
        if self.gram is not None:
            # среднее векторов n-грамм слова (без нулей)
            cnt = (g > 0).sum(-1, keepdim=True).clamp(min=1)
            out = out + self.gram(g).sum(-2) / cnt
        return out                              # [B, L, d]
# fmt: on


# fmt: off
def masked_max(h, mask):
    """Max-пулинг только по настоящим токенам (без нулей)."""
    h = h.masked_fill(~mask.unsqueeze(1), float('-inf'))
    return h.max(dim=2).values
# fmt: on


def masked_mean(h, mask):
    """Среднее по настоящим токенам."""
    m = mask.unsqueeze(1).float()
    return (h * m).sum(2) / m.sum(2).clamp(min=1)


# fmt: off
class TextCNN(nn.Module):
    """Свёрточная сеть для текста: эмбеддинги -> свёртки разной
    ширины -> max-пулинг -> полносвязные слои.

    kernel_sizes - ширина окон свёрток (по ветви на ширину);
    filters      - число фильтров (нейронов) в ветви;
    conv_layers  - число свёрточных слоёв в ветви;
    residual     - обходные соединения: h = f(conv(h)) + h;
    hidden       - размеры скрытых полносвязных слоёв.
    """

    def __init__(self, vocab_size, emb_dim=64,
                 kernel_sizes=(2, 3, 4), filters=64,
                 conv_layers=1, residual=False, hidden=(),
                 activation='relu', dropout=0.5, n_classes=2,
                 emb_mode='word', buckets=0, pooling='max',
                 emb_dropout=0.0):
        super().__init__()
        act = ACTIVATIONS[activation]
        self.emb = TokenEmbedding(vocab_size, emb_dim,
                                  emb_mode, buckets)
        self.emb_drop = nn.Dropout(emb_dropout)
        self.pooling = pooling
        self.residual = residual
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            layers = nn.ModuleList()
            for i in range(conv_layers):
                d_in = emb_dim if i == 0 else filters
                layers.append(nn.Conv1d(d_in, filters, k,
                                        padding='same'))
            self.branches.append(layers)
        self.act = act()
        size = filters * len(kernel_sizes)
        if pooling == 'maxmean':
            size *= 2
        self.out_size = size
        head = []
        for h in hidden:
            head += [nn.Dropout(dropout), nn.Linear(size, h),
                     act()]
            size = h
        head += [nn.Dropout(dropout),
                 nn.Linear(size, n_classes)]
        self.head = nn.Sequential(*head)

    def features(self, x, g=None):
        mask = x != 0
        e = self.emb_drop(self.emb(x, g))
        e = e.transpose(1, 2)          # [B, d, L]
        pooled = []
        for layers in self.branches:
            h = e
            for i, conv in enumerate(layers):
                z = self.act(conv(h))
                h = z + h if (self.residual and i > 0) else z
            pooled.append(masked_max(h, mask))
            if self.pooling == 'maxmean':
                pooled.append(masked_mean(h, mask))
        return torch.cat(pooled, dim=1)  # [B, F * ветви]

    def forward(self, x, g=None):
        return self.head(self.features(x, g))  # [B, 2]
# fmt: on


class BiGRUClassifier(nn.Module):
    """Рекуррентная сеть для сравнения: двунаправленный GRU и max-пулинг по выходам."""

    def __init__(
        self,
        vocab_size,
        emb_dim=64,
        hidden_size=64,
        dropout=0.5,
        n_classes=2,
        emb_mode='word',
        buckets=0,
    ):
        super().__init__()
        self.emb = TokenEmbedding(vocab_size, emb_dim, emb_mode, buckets)
        self.rnn = nn.GRU(emb_dim, hidden_size, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden_size, n_classes))

    def forward(self, x, g=None):
        mask = x != 0
        h, _ = self.rnn(self.emb(x, g))  # [B, L, 2H]
        return self.head(masked_max(h.transpose(1, 2), mask))


class CNNGRU(nn.Module):
    """Свёрточно-рекуррентная сеть: общие эмбеддинги, параллельно свёрточные ветви (локальные
    сочетания слов) и двунаправленный GRU (порядок слов во всём комментарии);
    признаки объединяются.
    """

    def __init__(
        self,
        vocab_size,
        emb_dim=64,
        kernel_sizes=(2, 3, 4),
        filters=64,
        hidden_size=64,
        dropout=0.5,
        n_classes=2,
        emb_mode='ngram',
        buckets=0,
    ):
        super().__init__()
        self.cnn = TextCNN(
            vocab_size, emb_dim, kernel_sizes, filters, emb_mode=emb_mode, buckets=buckets
        )
        self.rnn = nn.GRU(emb_dim, hidden_size, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.cnn.out_size + 2 * hidden_size, n_classes)
        )

    def forward(self, x, g=None):
        mask = x != 0
        conv = self.cnn.features(x, g)
        h, _ = self.rnn(self.cnn.emb(x, g))
        rec = masked_max(h.transpose(1, 2), mask)
        return self.head(torch.cat([conv, rec], dim=1))


def build(cfg, vocab_size):
    """Создаёт модель по словарю настроек (используется в экспериментах и в predict.py)."""
    cfg = dict(cfg)
    arch = cfg.pop('arch', 'cnn')
    keep = {
        'cnn': (
            'emb_dim',
            'kernel_sizes',
            'filters',
            'conv_layers',
            'residual',
            'hidden',
            'activation',
            'dropout',
            'emb_mode',
            'buckets',
            'pooling',
            'emb_dropout',
        ),
        'gru': ('emb_dim', 'hidden_size', 'dropout', 'emb_mode', 'buckets'),
        'cnngru': (
            'emb_dim',
            'kernel_sizes',
            'filters',
            'hidden_size',
            'dropout',
            'emb_mode',
            'buckets',
        ),
    }[arch]
    kw = {k: v for k, v in cfg.items() if k in keep}
    for k in ('kernel_sizes', 'hidden'):
        if k in kw:
            kw[k] = tuple(kw[k])
    cls = {'cnn': TextCNN, 'gru': BiGRUClassifier, 'cnngru': CNNGRU}[arch]
    return cls(vocab_size, **kw)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def macs_per_token(emb_dim, kernel_sizes, filters, conv_layers, hidden=(), n_classes=2):
    """Число умножений со сложением на одну позицию текста в свёрточной части
    плюс операции полносвязной части (на один пример), без учёта эмбеддингов."""
    conv = 0
    for k in kernel_sizes:
        for i in range(conv_layers):
            d_in = emb_dim if i == 0 else filters
            conv += filters * k * d_in
    fc, size = 0, filters * len(kernel_sizes)
    for h in hidden:
        fc += size * h
        size = h
    fc += size * n_classes
    return conv, fc

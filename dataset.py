# -*- coding: utf-8 -*-
"""Загрузка набора данных, очистка, разбиение на выборки и объекты-загрузчики PyTorch."""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import textprep as tp

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / 'data' / 'ru_toxicity.csv'
# исходный набор (часть ru):
# https://huggingface.co/datasets/textdetox/multilingual_toxicity_dataset
DATA_URL = (
    'https://huggingface.co/datasets/textdetox/multilingual_toxicity_dataset/'
    'resolve/main/data/ru-00000-of-00001.parquet'
)
MAX_LEN = 40  # самый длинный комментарий набора - 43 слова, длиннее 40 токенов - единицы


def load_frame(path=DATA_FILE):
    """Читает CSV (колонки text, toxic), удаляет пустые строки и повторы текстов."""
    df = pd.read_csv(path)
    df['text'] = df['text'].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
    df = df[df['text'].str.len() > 0]
    before = len(df)
    df = df.drop_duplicates(subset='text').reset_index(drop=True)
    df.attrs['duplicates'] = before - len(df)
    return df


def split_frame(df, seed=42):
    """Стратифицированное разбиение 70 / 15 / 15: доля токсичных одинакова во всех выборках."""
    train, rest = train_test_split(df, test_size=0.30, stratify=df['toxic'], random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest['toxic'], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


class Encoder:
    """Векторизация: текст -> индексы слов (и номера символьных n-грамм для гибридной модели)."""

    def __init__(
        self, train_texts, normalize=False, min_freq=1, buckets=0, max_grams=24, ngram_range=(3, 5)
    ):
        self.normalize = normalize
        self.buckets = buckets
        self.max_grams = max_grams
        self.ngram_range = tuple(ngram_range)
        self.vocab = tp.Vocab([self.tokens(t) for t in train_texts], min_freq=min_freq)

    def tokens(self, text):
        toks = tp.tokenize(text)[:MAX_LEN]
        return tp.normalize_tokens(toks) if self.normalize else toks

    def encode(self, text):
        toks = self.tokens(text) or ['<unk>']
        ids = self.vocab.encode(toks)
        grams = (
            tp.encode_ngrams(toks, self.buckets, self.max_grams, *self.ngram_range)
            if self.buckets
            else None
        )
        return ids, grams


class CommentsDataset(Dataset):
    """Набор комментариев: хранит уже закодированные тексты и метки."""

    def __init__(self, texts, labels, encoder):
        self.texts = list(texts)
        self.labels = [int(y) for y in labels]
        self.items = [encoder.encode(t) for t in self.texts]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        ids, grams = self.items[i]
        return ids, grams, self.labels[i]


# fmt: off
def collate(batch):
    """Собирает пакет: последовательности дополняются нулями
    до длины самой длинной в пакете."""
    n = max(len(ids) for ids, _, _ in batch)
    x = torch.zeros(len(batch), n, dtype=torch.long)
    g = None
    if batch[0][1] is not None:
        m = len(batch[0][1][0])
        g = torch.zeros(len(batch), n, m, dtype=torch.long)
    for i, (ids, grams, _) in enumerate(batch):
        x[i, :len(ids)] = torch.tensor(ids)
        if g is not None:
            g[i, :len(grams)] = torch.tensor(grams)
    y = torch.tensor([lab for _, _, lab in batch])
    return x, g, y
# fmt: on


def make_loaders(
    train_df, val_df, test_df, encoder, batch_size=64, seed=0, train_texts=None, train_labels=None
):
    """Три загрузчика. Обучающий перемешивает примеры в каждой эпохе (с фиксированным зерном)."""
    gen = torch.Generator().manual_seed(seed)
    tr_texts = train_df['text'] if train_texts is None else train_texts
    tr_labels = train_df['toxic'] if train_labels is None else train_labels
    train_ds = CommentsDataset(tr_texts, tr_labels, encoder)
    val_ds = CommentsDataset(val_df['text'], val_df['toxic'], encoder)
    test_ds = CommentsDataset(test_df['text'], test_df['toxic'], encoder)
    return (
        DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, generator=gen
        ),
        DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate),
        DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate),
    )


def augment(train_df, p=0.5, copies=1, seed=0):
    """Добавляет к обучающей выборке искажённые копии комментариев
    (замена букв, звёздочки, повторы)."""
    rng = random.Random(seed)
    texts, labels = list(train_df['text']), list(train_df['toxic'])
    for _ in range(copies):
        for t, y in zip(train_df['text'], train_df['toxic']):
            texts.append(tp.obfuscate(t, p, rng))
            labels.append(y)
    return texts, labels


def obfuscated_copy(df, p=0.5, seed=123):
    """Тестовая выборка с искажёнными словами - проверка устойчивости модели."""
    rng = random.Random(seed)
    out = df.copy()
    out['text'] = [tp.obfuscate(t, p, rng) for t in df['text']]
    return out


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

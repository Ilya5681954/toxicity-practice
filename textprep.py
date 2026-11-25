# -*- coding: utf-8 -*-
"""Предобработка текста: нормализация, токенизация, словарь, символьные n-граммы, искажение слов.

Модуль не зависит от PyTorch: его функции превращают строку комментария в списки целых чисел,
которые затем упаковываются в тензоры в модуле dataset.py.
"""

import random
import re
import zlib
from collections import Counter

PAD, UNK = 0, 1

# слово (кириллица, латиница, цифры и символы, которыми маскируют буквы) или одиночный знак/эмодзи
TOKEN_RE = re.compile(r"[а-яёa-z0-9*@$]+|[^\w\s]", re.IGNORECASE)
CYR_RE = re.compile(r"[а-яё]")

# латинские буквы и цифры, похожие по начертанию на русские буквы
LOOKALIKE = str.maketrans(
    {
        'a': 'а',
        'b': 'в',
        'c': 'с',
        'e': 'е',
        'h': 'н',
        'k': 'к',
        'm': 'м',
        'o': 'о',
        'p': 'р',
        't': 'т',
        'x': 'х',
        'y': 'у',
        'u': 'и',
        '0': 'о',
        '3': 'з',
        '4': 'ч',
        '6': 'б',
        '@': 'а',
        '$': 'с',
    }
)
REPEAT_RE = re.compile(r"(\w)\1{2,}")


def tokenize(text):
    """Разбивает текст на токены: слова и отдельные знаки (в том числе эмодзи)."""
    return TOKEN_RE.findall(text.lower())


def normalize_token(tok):
    """Приводит слово к «чистому» виду: ё -> е, латиница и цифры
    внутри русского слова -> кириллица,
    три и более одинаковых буквы подряд -> одна, звёздочки внутри слова удаляются."""
    tok = tok.replace('ё', 'е')
    if CYR_RE.search(tok):
        tok = tok.translate(LOOKALIKE)
        tok = tok.replace('*', '')
    tok = REPEAT_RE.sub(r"\1", tok)
    return tok or '*'


def normalize_tokens(tokens):
    return [normalize_token(t) for t in tokens]


class Vocab:
    """Словарь слов обучающей выборки: слово -> индекс (0 - заполнитель, 1 - неизвестное слово)."""

    def __init__(self, token_lists, min_freq=1):
        counts = Counter(t for toks in token_lists for t in toks)
        self.itos = ['<pad>', '<unk>'] + sorted(t for t, c in counts.items() if c >= min_freq)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.counts = counts

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(t, UNK) for t in tokens]

    def oov_rate(self, token_lists):
        total = sum(len(t) for t in token_lists)
        unknown = sum(1 for toks in token_lists for t in toks if t not in self.stoi)
        return unknown / max(total, 1)


def char_ngrams(token, n_min=3, n_max=5):
    """Символьные n-граммы слова с метками начала и конца: «<дура», «ура>» и т. д."""
    w = '<' + token + '>'
    grams = []
    for n in range(n_min, n_max + 1):
        grams.extend(w[i : i + n] for i in range(len(w) - n + 1))
    return grams or [w]


def hash_ngram(gram, buckets):
    """Номер корзины для n-граммы. crc32 даёт одинаковый результат при каждом запуске
    (встроенный hash() в Python от запуска к запуску меняется)."""
    return zlib.crc32(gram.encode('utf-8')) % (buckets - 1) + 1


def encode_ngrams(tokens, buckets, max_grams=24, n_min=3, n_max=5):
    """Для каждого токена - список номеров его n-грамм фиксированной
    длины max_grams (0 - заполнитель)."""
    out = []
    for t in tokens:
        ids = [hash_ngram(g, buckets) for g in char_ngrams(t, n_min, n_max)][:max_grams]
        out.append(ids + [0] * (max_grams - len(ids)))
    return out


# ------------------------------------------------------------------ искажение слов (аугментация)
TO_LOOKALIKE = {
    'о': ['0', 'o'],
    'а': ['a', '@'],
    'е': ['e'],
    'с': ['c'],
    'р': ['p'],
    'х': ['x'],
    'у': ['y'],
    'з': ['3'],
    'к': ['k'],
    'и': ['u'],
}
VOWELS = set('аеиоуыэюя')


def obfuscate_word(word, rng):
    """Одно случайное искажение слова так, как это делают авторы комментариев, обходя фильтры."""
    letters = [i for i, ch in enumerate(word) if CYR_RE.match(ch)]
    if len(letters) < 3:
        return word
    kind = rng.random()
    if kind < 0.4:  # буква -> похожий латинский символ или цифра
        cand = [i for i in letters if word[i] in TO_LOOKALIKE]
        if cand:
            i = rng.choice(cand)
            return word[:i] + rng.choice(TO_LOOKALIKE[word[i]]) + word[i + 1 :]
    if kind < 0.75:  # гласная внутри слова -> звёздочка
        cand = [i for i in letters[1:-1] if word[i] in VOWELS]
        if cand:
            i = rng.choice(cand)
            return word[:i] + '*' + word[i + 1 :]
    i = rng.choice(letters)  # растягивание: буква повторяется 3-4 раза
    return word[:i] + word[i] * rng.randint(3, 4) + word[i + 1 :]


def obfuscate(text, p, rng):
    """Искажает каждое слово длиной от 4 букв с вероятностью p."""
    parts = re.split(r"(\s+)", text)
    return ''.join(
        obfuscate_word(w, rng) if len(w) >= 4 and rng.random() < p else w for w in parts
    )


# ------------------------------------------------------------------ маскирование для отчёта
OBSCENE_ROOTS = (
    'хуй',
    'хуе',
    'хуё',
    'хуя',
    'хуи',
    'пизд',
    'еба',
    'ебл',
    'ебу',
    'ебн',
    'еби',
    'ёба',
    'бля',
    'сук',
    'муда',
    'мудо',
    'пидор',
    'пидр',
    'залуп',
    'гандон',
    'шлюх',
    'манда',
)


def is_obscene(word):
    w = word.lower().replace('ё', 'е')
    return any(r.replace('ё', 'е') in w for r in OBSCENE_ROOTS)


def mask_word(word):
    """Маскирует бранное слово для отчёта: первая и последняя буквы, между ними звёздочки."""
    if len(word) <= 2 or not is_obscene(word):
        return word
    return word[0] + '*' * (len(word) - 2) + word[-1]


def mask_text(text):
    return re.sub(r"[а-яёa-z]+", lambda m: mask_word(m.group(0)), text, flags=re.IGNORECASE)


def demo():
    rng = random.Random(0)
    s = 'Ну ты и дуууурак, честное слово!'
    print(tokenize(s))
    print(normalize_tokens(tokenize(obfuscate(s.lower(), 1.0, rng))))
    print(char_ngrams('дурак'))


if __name__ == '__main__':
    demo()

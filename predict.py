# -*- coding: utf-8 -*-
"""Проверка комментариев обученной моделью (results/model.pt создаёт main.py).

python predict.py "Спасибо, очень полезная статья" "ты тупой идиот"
python predict.py < comments.txt          # по одному комментарию в строке
"""

import sys
from pathlib import Path

import torch

import dataset as ds
import models
import textprep as tp

HERE = Path(__file__).resolve().parent


class Moderator:
    def __init__(self, path=HERE / 'results' / 'model.pt'):
        pack = torch.load(path, map_location='cpu', weights_only=False)
        cfg = pack['cfg']
        self.threshold = pack['threshold']
        self.encoder = ds.Encoder(
            [], normalize=cfg['normalize'], buckets=cfg['buckets'], ngram_range=cfg['ngram_range']
        )
        self.encoder.vocab.itos = pack['itos']
        self.encoder.vocab.stoi = {t: i for i, t in enumerate(pack['itos'])}
        self.model = models.build(cfg, len(pack['itos']))
        self.model.load_state_dict(pack['state_dict'])
        self.model.eval()

    @torch.no_grad()
    def score(self, texts):
        batch = [(*self.encoder.encode(t), 0) for t in texts]
        x, g, _ = ds.collate(batch)
        return torch.softmax(self.model(x, g), dim=1)[:, 1].tolist()


def main():
    texts = sys.argv[1:] or [line.strip() for line in sys.stdin if line.strip()]
    if not texts:
        print(__doc__)
        return
    mod = Moderator()
    for t, p in zip(texts, mod.score(texts)):
        verdict = 'ТОКСИЧНЫЙ' if p >= mod.threshold else 'допустимый'
        print('%5.1f %%  %-10s  %s' % (100 * p, verdict, tp.mask_text(t)))


if __name__ == '__main__':
    main()

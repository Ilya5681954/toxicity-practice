# -*- coding: utf-8 -*-
"""Серии экспериментов, результаты которых приведены в отчёте.

Каждый вариант сети обучается с тремя зёрнами генератора случайных чисел (0, 1, 2), выбор делается
по точности на проверочной (валидационной) выборке. Результаты каждого запуска сохраняются
в results/runs_<серия>.json; при повторном запуске уже посчитанные варианты не обучаются заново.

    python experiments.py topology          # 2.2.1 выбор топологии
    python experiments.py kernels|filters|layers|activation|hidden   # 2.2.3 подбор характеристик
    python experiments.py timing            # 2.2.3 время эпохи в зависимости от топологии
    python experiments.py hyper_opt|hyper_reg|hyper_ngram|data|arch   # 2.4.4 улучшения
    python experiments.py all               # всё по порядку (около полутора часов на 4 ядрах)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import dataset as ds
import models
import training

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
RES.mkdir(exist_ok=True)
SEEDS = (0, 1, 2)

# первый вариант сети (раздел 1.3): слова, одна ветвь свёртки шириной 3
V1 = dict(
    arch='cnn',
    emb_mode='word',
    emb_dim=64,
    kernel_sizes=[3],
    filters=64,
    conv_layers=1,
    residual=False,
    hidden=[],
    activation='relu',
    dropout=0.5,
    buckets=0,
)
# база для подбора характеристик (раздел 2.2.3): представление слов символьными n-граммами
BASE = dict(V1, emb_mode='ngram', kernel_sizes=[2, 3, 4], buckets=50000)
TRAIN = dict(
    lr=1e-3,
    batch=64,
    weight_decay=0.0,
    scheduler=False,
    epochs=30,
    patience=4,
    normalize=False,
    ngram_range=[3, 5],
    augment=0.0,
)

_ENC, _DATA = {}, {}


def data():
    if not _DATA:
        df = ds.load_frame()
        tr, va, te = ds.split_frame(df)
        _DATA.update(
            df=df,
            train=tr,
            val=va,
            test=te,
            val_obf=ds.obfuscated_copy(va, 0.5, seed=7),
            test_obf=ds.obfuscated_copy(te, 0.5, seed=123),
        )
    return _DATA


def full(cfg):
    out = dict(TRAIN)
    out.update(BASE if cfg.get('base', 'ngram') == 'ngram' else V1)
    out.update({k: v for k, v in cfg.items() if k != 'base'})
    return out


def key_of(cfg, seed):
    return json.dumps({**cfg, 'seed': seed}, sort_keys=True, ensure_ascii=False)


def encoder(cfg):
    k = (cfg['normalize'], cfg['buckets'], tuple(cfg['ngram_range']))
    if k not in _ENC:
        d = data()
        _ENC[k] = ds.Encoder(
            d['train']['text'],
            normalize=cfg['normalize'],
            buckets=cfg['buckets'],
            ngram_range=cfg['ngram_range'],
        )
    return _ENC[k]


def run_once(cfg, seed, keep_model=False):
    d = data()
    enc = encoder(cfg)
    ds.set_seed(seed)
    texts = labels = None
    if cfg['augment']:
        texts, labels = ds.augment(d['train'], p=cfg['augment'], copies=1, seed=seed)
    tr_l, va_l, te_l = ds.make_loaders(
        d['train'],
        d['val'],
        d['test'],
        enc,
        cfg['batch'],
        seed,
        train_texts=texts,
        train_labels=labels,
    )
    model = models.build(cfg, len(enc.vocab))
    t0 = time.perf_counter()
    hist = training.train(
        model,
        tr_l,
        va_l,
        epochs=cfg['epochs'],
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay'],
        patience=cfg['patience'],
        scheduler=cfg['scheduler'],
    )
    total = time.perf_counter() - t0
    res = training.summarize(hist)
    res.update(seed=seed, total_time=total, params=models.count_params(model), history=hist)
    vo = ds.make_loaders(d['train'], d['val_obf'], d['test_obf'], enc, cfg['batch'], seed)
    p, y, _ = training.predict_proba(model, vo[1])
    res['val_obf_acc'] = training.metrics(y, p)['accuracy']
    p, y, _ = training.predict_proba(model, va_l)
    res['val'] = training.metrics(y, p)
    p, y, _ = training.predict_proba(model, te_l)
    res['test'] = training.metrics(y, p)
    p, y, _ = training.predict_proba(model, vo[2])
    res['test_obf'] = training.metrics(y, p)
    if keep_model:
        res['model'], res['encoder'] = model, enc
    return res


def _cached(key):
    for p in RES.glob('runs_*.json'):
        try:
            store = json.loads(p.read_text(encoding='utf-8'))
        except ValueError:
            continue
        if key in store:
            return store[key]
    return None


def run(series, name, cfg, seeds=SEEDS, log=True):
    """Обучает вариант cfg с каждым зерном; уже посчитанное берёт из файла серии."""
    cfg = full(cfg)
    path = RES / ('runs_%s.json' % series)
    store = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    rows = []
    for s in seeds:
        k = key_of(cfg, s)
        if k not in store:
            cached = _cached(k)  # тот же вариант мог быть обучен в другой серии
            if cached is not None:
                store[k] = dict(cached, name=name)
            else:
                r = run_once(cfg, s)
                store[k] = {'name': name, 'cfg': cfg, **r}
            path.write_text(json.dumps(store, ensure_ascii=False, indent=1), encoding='utf-8')
        rows.append(store[k])
    acc = [r['val_acc'] for r in rows]
    summary = {
        'name': name,
        'cfg': cfg,
        'val_acc': float(np.mean(acc)),
        'val_acc_std': float(np.std(acc)),
        'val_obf_acc': float(np.mean([r['val_obf_acc'] for r in rows])),
        'val_f1': float(np.mean([r['val']['f1'] for r in rows])),
        'epoch_time': float(np.mean([r['epoch_time'] for r in rows])),
        'best_epoch': float(np.mean([r['best_epoch'] for r in rows])),
        'params': rows[0]['params'],
    }
    if log:
        print(
            '%-28s точность %.4f ± %.4f  (искаж. %.4f)  эпоха %.2f с  параметров %d'
            % (
                name,
                summary['val_acc'],
                summary['val_acc_std'],
                summary['val_obf_acc'],
                summary['epoch_time'],
                summary['params'],
            ),
            flush=True,
        )
    return summary


# ------------------------------------------------------------------ серии
TOPOLOGY = [
    ('Вариант 1: слова, k=3', dict(base='v1')),
    ('Слова, k=2,3,4', dict(base='v1', kernel_sizes=[2, 3, 4])),
    ('Слова, BiGRU', dict(base='v1', arch='gru', hidden_size=64)),
    ('n-граммы, k=2,3,4', dict()),
    ('n-граммы, BiGRU', dict(arch='gru', hidden_size=64)),
    ('Слова + n-граммы, k=2,3,4', dict(emb_mode='hybrid')),
]
GROUPS = {
    'kernels': [
        ('k = 3', dict(kernel_sizes=[3])),
        ('k = 2, 3', dict(kernel_sizes=[2, 3])),
        ('k = 2, 3, 4', dict()),
        ('k = 2, 3, 4, 5', dict(kernel_sizes=[2, 3, 4, 5])),
        ('k = 1, 2, 3', dict(kernel_sizes=[1, 2, 3])),
    ],
    'filters': [('F = %d' % f, dict(filters=f)) for f in (16, 32, 64, 128, 256)],
    'layers': [
        ('1 слой', dict()),
        ('2 слоя', dict(conv_layers=2)),
        ('2 слоя + обход', dict(conv_layers=2, residual=True)),
        ('3 слоя', dict(conv_layers=3)),
        ('3 слоя + обход', dict(conv_layers=3, residual=True)),
    ],
    'activation': [
        (a, dict(activation=a)) for a in ('relu', 'leaky_relu', 'gelu', 'tanh', 'sigmoid')
    ],
    'hidden': [
        ('без скрытого слоя', dict()),
        ('скрытый 64', dict(hidden=[64])),
        ('скрытые 128, 64', dict(hidden=[128, 64])),
    ],
}


def final_cfg():
    """Окончательная конфигурация после подбора характеристик (см. results/final_config.json)."""
    path = RES / 'final_config.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(BASE)


def improvement_groups():
    f = final_cfg()
    f.pop('base', None)
    ref = ('исходные параметры', dict(f))
    return {
        'hyper_opt': [
            ref,
            ('lr = 0,003', dict(f, lr=3e-3)),
            ('lr = 0,0003', dict(f, lr=3e-4)),
            ('пакет 32', dict(f, batch=32)),
            ('пакет 128', dict(f, batch=128)),
            ('снижение lr на плато', dict(f, scheduler=True)),
        ],
        'hyper_reg': [
            ref,
            ('dropout 0,3', dict(f, dropout=0.3)),
            ('dropout 0,7', dict(f, dropout=0.7)),
            ('эмбеддинг 32', dict(f, emb_dim=32)),
            ('эмбеддинг 128', dict(f, emb_dim=128)),
            ('L2-регуляризация 1e-5', dict(f, weight_decay=1e-5)),
        ],
        'hyper_ngram': [
            ref,
            ('n-граммы 2-4', dict(f, ngram_range=[2, 4])),
            ('n-граммы 3-6', dict(f, ngram_range=[3, 6])),
            ('корзин 20 000', dict(f, buckets=20000)),
            ('корзин 100 000', dict(f, buckets=100000)),
        ],
        'data': [
            ('исходные данные', dict(f)),
            ('нормализация', dict(f, normalize=True)),
            ('аугментация', dict(f, augment=0.5)),
            ('нормализация + аугментация', dict(f, normalize=True, augment=0.5)),
        ],
        'arch': [
            ('исходная архитектура', dict(f)),
            ('пулинг max + mean', dict(f, pooling='maxmean')),
            ('dropout эмбеддингов 0,2', dict(f, emb_dropout=0.2)),
            ('свёртки + BiGRU', dict(f, arch='cnngru', hidden_size=64)),
        ],
    }


IMPROVE = ('hyper_opt', 'hyper_reg', 'hyper_ngram', 'data', 'arch')


def timing():
    """Время эпохи для сетей разного размера: один запуск на 3 эпохи
    без ранней остановки, 4 потока."""
    torch.set_num_threads(4)
    d = data()
    out = []
    variants = [('Вариант 1 (слова)', dict(base='v1'))]
    for f in (16, 32, 64, 128, 256):
        variants.append(('F=%d' % f, dict(filters=f)))
    for L in (2, 3):
        variants.append(('%d слоя, F=64' % L, dict(conv_layers=L)))
        variants.append(('%d слоя, F=128' % L, dict(conv_layers=L, filters=128)))
    for ks in ([3], [2, 3], [2, 3, 4, 5]):
        variants.append(('k=' + ','.join(map(str, ks)), dict(kernel_sizes=ks)))
    variants.append(('скрытые 128,64', dict(hidden=[128, 64])))
    variants.append(('эмбеддинг 128', dict(emb_dim=128)))
    for name, c in variants:
        cfg = full(c)
        enc = encoder(cfg)
        ds.set_seed(0)
        tr_l, va_l, _ = ds.make_loaders(d['train'], d['val'], d['test'], enc, cfg['batch'], 0)
        model = models.build(cfg, len(enc.vocab))
        hist = training.train(model, tr_l, va_l, epochs=3, patience=99)
        conv, fc = models.macs_per_token(
            cfg['emb_dim'], cfg['kernel_sizes'], cfg['filters'], cfg['conv_layers'], cfg['hidden']
        )
        row = {
            'name': name,
            'cfg': cfg,
            'params': models.count_params(model),
            'conv_macs': conv,
            'fc_macs': fc,
            'epoch_time': float(np.median([r['time'] for r in hist])),
        }
        out.append(row)
        print(
            '%-18s параметров %8d  MAC/позиция %7d  эпоха %.2f с'
            % (name, row['params'], conv, row['epoch_time']),
            flush=True,
        )
    (RES / 'timing.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8'
    )


def _load(series):
    return json.loads((RES / ('summary_%s.json' % series)).read_text(encoding='utf-8'))


def select_final():
    """Окончательная конфигурация: в каждой серии 2.2.3 берётся вариант
    с наибольшей средней точностью,
    найденные значения объединяются; объединённый вариант проверяется тремя запусками."""
    cfg = full({})
    chosen = {}
    for g, variants in GROUPS.items():
        rows = _load(g)
        best = max(rows, key=lambda r: r['val_acc'])
        change = dict(variants[[r['name'] for r in rows].index(best['name'])][1])
        chosen[g] = best['name']
        cfg.update(change)
    combined = run('final', 'итоговая конфигурация', cfg)
    base = run('final', 'исходная база', {})
    if combined['val_acc'] < base['val_acc']:
        cfg = full({})
    (RES / 'final_config.json').write_text(
        json.dumps(cfg, ensure_ascii=False, indent=1), encoding='utf-8'
    )
    (RES / 'summary_final.json').write_text(
        json.dumps([base, combined], ensure_ascii=False, indent=1), encoding='utf-8'
    )
    print('выбрано:', chosen)


def score(row):
    """Критерий выбора улучшений: среднее точности на исходной
    и на искажённой проверочных выборках.
    В реальных комментариях часть бранных слов намеренно искажена, поэтому важны оба показателя."""
    return (row['val_acc'] + row['val_obf_acc']) / 2


def select_improved():
    """Улучшенная конфигурация: в каждой серии 2.4.4 берётся лучший по критерию score() вариант,
    если он лучше исходного; найденные изменения объединяются и проверяются тремя запусками."""
    groups = improvement_groups()
    ref_cfg = final_cfg()
    gains = []
    for g in IMPROVE:
        rows = _load(g)
        best = max(rows, key=score)
        if score(best) > score(rows[0]):
            change = dict(groups[g][[r['name'] for r in rows].index(best['name'])][1])
            gains.append((g, best['name'], score(best) - score(rows[0]), change))
    cfg = dict(ref_cfg)
    for g, name, gain, change in gains:
        cfg.update({k: v for k, v in change.items() if v != ref_cfg.get(k)})
    ref = run('improved', 'окончательная сеть (2.2)', ref_cfg)
    combined = run('improved', 'все улучшения вместе', cfg)
    singles = [
        (run('improved', name, dict(ref_cfg, **change)), dict(ref_cfg, **change))
        for g, name, gain, change in gains
    ]
    cands = [(combined, cfg)] + singles
    best_row, best_cfg = max(cands, key=lambda c: score(c[0]))
    out = best_cfg if score(best_row) > score(ref) else ref_cfg
    (RES / 'improved_config.json').write_text(
        json.dumps(full(out), ensure_ascii=False, indent=1), encoding='utf-8'
    )
    rows = [ref] + [c[0] for c in singles] + [combined]
    for row in rows:
        row['score'] = score(row)
    (RES / 'summary_improved.json').write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8'
    )
    print('улучшения:', [(g, n, round(x, 4)) for g, n, x, _ in gains], 'выбран:', best_row['name'])


def run_series(series):
    torch.set_num_threads(4 if series.startswith('select') else 2)
    if series == 'select_final':
        return select_final()
    if series == 'select_improved':
        return select_improved()
    if series == 'topology':
        rows = [run(series, n, c) for n, c in TOPOLOGY]
    elif series in GROUPS:
        rows = [run(series, n, c) for n, c in GROUPS[series]]
    elif series in IMPROVE:
        rows = [run(series, n, c) for n, c in improvement_groups()[series]]
    elif series == 'timing':
        return timing()
    else:
        raise SystemExit('неизвестная серия: ' + series)
    (RES / ('summary_%s.json' % series)).write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8'
    )


if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else 'all'
    order = (
        ['topology', *GROUPS, 'select_final', *IMPROVE, 'select_improved', 'timing']
        if arg == 'all'
        else [arg]
    )
    for s in order:
        print('==== серия', s, flush=True)
        run_series(s)

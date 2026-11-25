# -*- coding: utf-8 -*-
"""Основной сценарий: данные -> первый вариант сети -> окончательная
и улучшенная сеть -> оценка -> графики.

    python main.py

Работает на процессоре около 5-10 минут. Результаты: results/report_data.json (все числа отчёта),
results/figures/*.png|svg (рисунки), results/model.pt (обученная модель для predict.py).
Серии экспериментов по подбору характеристик запускаются отдельно: python experiments.py all.
"""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import dataset as ds
import experiments as ex
import models
import neurons
import plots
import textprep as tp
import training

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'

# контрольные фразы для проверки адекватности (придуманы автором, в наборе данных их нет)
CONTROL = [
    ('Спасибо за помощь, всё получилось с первого раза', 0),
    ('Не согласен с автором, но статья интересная', 0),
    ('Дорогу опять не почистили, жители недовольны', 0),
    ('Этот фильм просто бомба, всем советую', 0),
    ('Купил хрен и свёклу для винегрета', 0),
    ('Наша сука ощенилась, все щенки здоровы', 0),
    ('Кто убил Кеннеди, до сих пор спорят историки', 0),
    ('Ты тупой идиот, закрой рот', 1),
    ('Автор дебил, пиши дальше свою чушь', 1),
    ('Да пошёл ты, клоун', 1),
    ('Какое же ты ничтожество', 1),
    ('Заткнись, урод', 1),
    ('ты ид*от и дyрак', 1),
    ('Депутаты опять наворовали, сволочи', 1),
]
OBF_WORDS = [
    ('дурак', ['дуррак', 'дyрак', 'д*рак', 'дураки', 'дурачок']),
    ('идиот', ['идиотт', 'ид*от', 'идиоты', 'идиотизм']),
    ('урод', ['урооод', 'yрод', 'уроды', 'уродина']),
]


def log_odds(train, top=10):
    """Слова, сильнее всего связанные с классом: логарифм отношения шансов со сглаживанием."""
    cnt = {0: Counter(), 1: Counter()}
    for t, y in zip(train['text'], train['toxic']):
        cnt[y].update(set(w for w in tp.tokenize(t) if w.isalpha() and len(w) > 2))
    n = {y: (train['toxic'] == y).sum() for y in (0, 1)}
    rows = []
    for w in set(cnt[0]) | set(cnt[1]):
        a, b = cnt[1][w], cnt[0][w]
        if a + b < 15:
            continue
        lo = math.log((a + 0.5) / (n[1] - a + 0.5)) - math.log((b + 0.5) / (n[0] - b + 0.5))
        rows.append((tp.mask_word(w), lo, a + b))
    rows.sort(key=lambda r: r[1])
    return rows[:top] + rows[-top:][::-1]


def coverage(train, val):
    enc_w = ds.Encoder(train['text'])
    enc_n = ds.Encoder(train['text'], normalize=True)
    toks = [enc_w.tokens(t) for t in val['text']]
    seen = set(g for t in train['text'] for w in tp.tokenize(t) for g in tp.char_ngrams(w))
    flat = [w for ts in toks for w in ts]
    no_gram = sum(1 for w in flat if not any(g in seen for g in tp.char_ngrams(w))) / len(flat)
    grams = [g for w in flat for g in tp.char_ngrams(w)]
    unseen_grams = sum(1 for g in grams if g not in seen) / len(grams)
    return {
        'vocab_words': len(enc_w.vocab) - 2,
        'vocab_words_norm': len(enc_n.vocab) - 2,
        'oov_words': enc_w.vocab.oov_rate(toks),
        'oov_words_norm': enc_n.vocab.oov_rate([enc_n.tokens(t) for t in val['text']]),
        'unseen_ngrams': unseen_grams,
        'tokens_without_known_ngram': no_gram,
        'ngrams_train': len(seen),
    }


def word_vector(model, enc, word):
    ids, grams = enc.encode(word)
    x = torch.tensor([ids])
    g = torch.tensor([grams]) if grams is not None else None
    with torch.no_grad():
        return model.emb(x, g)[0, 0].numpy()


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def train_named(cfg, seed=0):
    r = ex.run_once(ex.full(cfg), seed, keep_model=True)
    return r


def evaluate_detail(r, d):
    model, enc = r['model'], r['encoder']
    loaders = ds.make_loaders(d['train'], d['val'], d['test'], enc)
    probs, labels, _ = training.predict_proba(model, loaders[2])
    return probs, labels


def main():
    torch.set_num_threads(4)
    out = {}
    # 1. контрольный пример математической модели
    m, _ = neurons.check()
    out['neuron'] = {k: v for k, v in m.items()}
    print('контрольный пример: ручной расчёт совпал с PyTorch, L =', round(m['loss'], 4))

    # 2. данные
    d = ex.data()
    df, tr, va, te = d['df'], d['train'], d['val'], d['test']
    words = df['text'].str.split().str.len()
    out['data'] = {
        'rows_raw': len(df) + df.attrs['duplicates'],
        'duplicates': df.attrs['duplicates'],
        'rows': len(df),
        'toxic': int(df['toxic'].sum()),
        'split': [len(tr), len(va), len(te)],
        'toxic_share': [float(x['toxic'].mean()) for x in (tr, va, te)],
        'words_mean': float(words.mean()),
        'words_median': float(words.median()),
        'words_max': int(words.max()),
        'words_by_class': [float(words[df['toxic'] == c].mean()) for c in (0, 1)],
        'chars_mean': float(df['text'].str.len().mean()),
        'coverage': coverage(tr, va),
    }
    cov = out['data']['coverage']
    plots.lengths(df)
    lo = log_odds(tr)
    out['data']['log_odds'] = lo
    plots.top_tokens(lo)
    plots.coverage(
        [
            ('словарь слов', cov['oov_words']),
            ('словарь слов после нормализации', cov['oov_words_norm']),
            ('символьные n-граммы (нет ни одной известной)', cov['tokens_without_known_ngram']),
        ]
    )
    print('данные:', out['data']['rows'], 'комментариев,', out['data']['split'])

    # 3. классические модели для сравнения
    base = {}
    for an, rng in (('word', (1, 1)), ('char_wb', (2, 5))):
        v = TfidfVectorizer(analyzer=an, ngram_range=rng, sublinear_tf=True)
        lr = LogisticRegression(C=10, max_iter=3000).fit(v.fit_transform(tr['text']), tr['toxic'])
        base[an] = {
            'val': lr.score(v.transform(va['text']), va['toxic']),
            'test': lr.score(v.transform(te['text']), te['toxic']),
            'test_obf': lr.score(v.transform(d['test_obf']['text']), d['test_obf']['toxic']),
        }
    out['baseline'] = base

    # 4. первый вариант сети, окончательная и улучшенная конфигурации (зерно 0)
    final = ex.final_cfg()
    improved_path = RES / 'improved_config.json'
    improved = (
        json.loads(improved_path.read_text(encoding='utf-8')) if improved_path.exists() else final
    )
    runs = {}
    for name, cfg in (('v1', dict(base='v1')), ('final', final), ('improved', improved)):
        r = train_named(cfg)
        runs[name] = r
        out[name] = {
            k: r[k]
            for k in (
                'history',
                'val',
                'test',
                'test_obf',
                'params',
                'best_epoch',
                'epoch_time',
                'epochs_run',
                'total_time',
                'val_obf_acc',
            )
        }
        out[name]['cfg'] = ex.full(cfg)
        print(
            '%-9s тест: точность %.4f, F1 %.4f, искажённый тест %.4f'
            % (name, r['test']['accuracy'], r['test']['f1'], r['test_obf']['accuracy'])
        )
    plots.curves(runs['v1']['history'], 'train_curves_v1')
    plots.curves(runs['final']['history'], 'train_curves_final')
    plots.curves(runs['improved']['history'], 'train_curves_improved')

    det = {}
    for name in ('v1', 'final', 'improved'):
        det[name] = evaluate_detail(runs[name], d)
    probs, labels = det['final']
    plots.confusion(runs['final']['test']['confusion'], 'eval_confusion_final')
    plots.confusion(runs['improved']['test']['confusion'], 'eval_confusion_improved')
    plots.roc_pr(
        [
            ('вариант 1 (слова)',) + det['v1'],
            ('окончательная сеть',) + det['final'],
            ('улучшенная сеть',) + det['improved'],
        ]
    )
    plots.threshold(labels, probs)
    plots.prob_hist(labels, probs)
    best_t = max(np.linspace(0.1, 0.9, 81), key=lambda t: training.metrics(labels, probs, t)['f1'])
    out['threshold'] = {
        'best_f1_t': float(best_t),
        'at_best': training.metrics(labels, probs, best_t),
        'at_0_3': training.metrics(labels, probs, 0.3),
        'at_0_5': training.metrics(labels, probs, 0.5),
        'at_0_7': training.metrics(labels, probs, 0.7),
    }
    # ошибки окончательной сети на тестовой выборке (для отчёта бранные слова маскируются)
    texts = list(te['text'])
    err = [
        (float(p), int(y), tp.mask_text(t))
        for p, y, t in zip(probs, labels, texts)
        if (p >= 0.5) != y
    ]
    err.sort(key=lambda e: -abs(e[0] - 0.5))
    out['errors'] = {
        'fp': [e for e in err if e[1] == 0][:10],
        'fn': [e for e in err if e[1] == 1][:10],
        'n_fp': sum(1 for e in err if e[1] == 0),
        'n_fn': sum(1 for e in err if e[1] == 1),
    }
    imp_p, imp_l = det['improved']
    out['fixed_by_improved'] = int(
        sum(1 for p, q, y in zip(probs, imp_p, labels) if (p >= 0.5) != y and (q >= 0.5) == y)
    )
    out['broken_by_improved'] = int(
        sum(1 for p, q, y in zip(probs, imp_p, labels) if (p >= 0.5) == y and (q >= 0.5) != y)
    )

    # 5. адекватность: контрольные фразы, похожесть искажённых слов
    def control_probs(name):
        mdl, en = runs[name]['model'], runs[name]['encoder']
        ld = torch.utils.data.DataLoader(
            ds.CommentsDataset([c[0] for c in CONTROL], [c[1] for c in CONTROL], en),
            batch_size=64,
            collate_fn=ds.collate,
        )
        return training.predict_proba(mdl, ld)[0]

    cps = {n: control_probs(n) for n in ('v1', 'final', 'improved')}
    out['control'] = [
        (txt, y, float(a), float(b), float(c))
        for (txt, y), a, b, c in zip(CONTROL, cps['v1'], cps['final'], cps['improved'])
    ]
    v1e = runs['v1']['encoder']
    sims = []
    for w, variants in OBF_WORDS:
        for v in variants:
            row = [w, v, v in v1e.vocab.stoi]
            for name in ('final', 'improved'):
                mdl, en = runs[name]['model'], runs[name]['encoder']
                mdl.eval()
                row.append(cosine(word_vector(mdl, en, w), word_vector(mdl, en, v)))
            sims.append(row)
    out['similarity'] = sims
    model, enc = runs['improved']['model'], runs['improved']['encoder']
    rob = [
        ('Вариант 1 (слова)', runs['v1']['test']['accuracy'], runs['v1']['test_obf']['accuracy']),
        ('TF-IDF + логистическая регрессия', base['char_wb']['test'], base['char_wb']['test_obf']),
        (
            'Окончательная сеть (2.2)',
            runs['final']['test']['accuracy'],
            runs['final']['test_obf']['accuracy'],
        ),
        (
            'Улучшенная сеть (2.4.4)',
            runs['improved']['test']['accuracy'],
            runs['improved']['test_obf']['accuracy'],
        ),
    ]
    plots.robustness(rob)
    out['robustness'] = rob

    # 6. графики по сериям экспериментов (если они уже посчитаны)
    def summ(name):
        p = RES / ('summary_%s.json' % name)
        return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None

    if summ('topology'):
        plots.bars(summ('topology'), 'exp_topology', target=0.7, xlim=(0.5, 0.97))
    groups = {k: summ(k) for k in ex.GROUPS if summ(k)}
    if groups:
        plots.group_grid(groups)
    tpath = RES / 'timing.json'
    if tpath.exists():
        rows = json.loads(tpath.read_text(encoding='utf-8'))
        # в аппроксимацию входят варианты с одинаковой таблицей эмбеддингов (50 000 x 64)
        fit_rows = [r for r in rows if r['cfg']['emb_mode'] == 'ngram' and r['cfg']['emb_dim'] == 64]
        x = np.array([r['conv_macs'] for r in fit_rows], dtype=float)
        t = np.array([r['epoch_time'] for r in fit_rows])
        b, a = np.polyfit(x, t, 1)
        r2 = 1 - ((t - (a + b * x)) ** 2).sum() / ((t - t.mean()) ** 2).sum()
        out['timing'] = {'rows': rows, 'a': float(a), 'b': float(b), 'r2': float(r2), 'n_fit': len(fit_rows)}
        plots.timing(fit_rows, (a, b))
    hyper = []
    for k in ('hyper_opt', 'hyper_reg', 'hyper_ngram'):
        rows = summ(k) or []
        hyper += rows if not hyper else rows[1:]
    if hyper:
        plots.bars(hyper, 'imp_hyper')
    if summ('data') and summ('arch'):
        rows = [(r['name'], r['val_acc'], r['val_obf_acc']) for r in summ('data')]
        rows += [(r['name'], r['val_acc'], r['val_obf_acc']) for r in summ('arch')[1:]]
        rows[0] = ('исходная конфигурация',) + rows[0][1:]
        plots.robustness(rows, 'imp_da', xlabel='Средняя точность на проверочной выборке (3 запуска)',
                         xlim=(0.75, 0.97), labels=('исходная проверочная выборка', 'искажённая проверочная выборка'))
    hp = RES / 'runs_hyper_opt.json'
    if hp.exists():
        store = json.loads(hp.read_text(encoding='utf-8'))
        pick = {}
        for v in store.values():
            if v['seed'] == 0 and v['name'] in ('lr = 0,003', 'исходные параметры', 'lr = 0,0003'):
                pick[v['name']] = v['history']
        order = [
            ('lr = 0,003', 'lr = 0,003'),
            ('исходные параметры', 'lr = 0,001'),
            ('lr = 0,0003', 'lr = 0,0003'),
        ]
        if len(pick) == 3:
            plots.lr_curves([(lab, pick[n]) for n, lab in order])

    # 7. модель для predict.py
    torch.save(
        {
            'state_dict': model.state_dict(),
            'cfg': ex.full(improved),
            'itos': enc.vocab.itos,
            'threshold': 0.5,
        },
        RES / 'model.pt',
    )
    # 8. быстродействие и пример работы predict.py
    import time
    import predict

    mod = predict.Moderator(RES / 'model.pt')
    mod.score(['прогрев'])
    t0 = time.perf_counter()
    for _ in range(200):
        mod.score(['Не согласен с автором, но статья интересная'])
    single = (time.perf_counter() - t0) / 200
    t0 = time.perf_counter()
    mod.score(list(te['text']))
    batch = time.perf_counter() - t0
    out['speed'] = {'single_ms': single * 1000, 'test_batch_s': batch, 'test_n': len(te)}
    demo = ['Спасибо, полезная статья', 'Не согласен, но спорить не буду', 'ты тупой идиот',
            'автор ид*от, пиши дальше', 'Заткнись, урод']
    lines = ['$ python predict.py "%s" \\' % demo[0], '    "%s" "%s" \\' % (demo[1], demo[2]),
             '    "%s" "%s"' % (demo[3], demo[4])]
    for txt, p in zip(demo, mod.score(demo)):
        verdict = 'ТОКСИЧНЫЙ' if p >= mod.threshold else 'допустимый'
        lines.append('%5.1f %%  %-10s  %s' % (100 * p, verdict, tp.mask_text(txt)))
    out['predict_demo'] = '\n'.join(lines)
    print(out['predict_demo'])
    print('один комментарий: %.2f мс' % (single * 1000))

    (RES / 'report_data.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding='utf-8'
    )
    print('готово: results/report_data.json, results/figures, results/model.pt')


if __name__ == '__main__':
    main()

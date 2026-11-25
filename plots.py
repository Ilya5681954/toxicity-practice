# -*- coding: utf-8 -*-
"""Графики для отчёта. Каждый рисунок сохраняется в PNG (растр)
и SVG (вектор) в results/figures."""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / 'results' / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

# цвета рядов в постоянном порядке: 1 - синий, 2 - оранжевый, 3 - бирюзовый
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
CLASS_NAMES = ['нетоксичный', 'токсичный']

plt.rcParams.update(
    {
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
        'font.size': 11,
        'axes.titlesize': 11,
        'axes.labelsize': 11,
        'axes.edgecolor': INK2,
        'axes.labelcolor': INK,
        'xtick.color': INK2,
        'ytick.color': INK2,
        'axes.grid': True,
        'grid.color': GRID,
        'grid.linewidth': 0.6,
        'axes.axisbelow': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
        'lines.linewidth': 2,
        'svg.fonttype': 'path',
        'figure.dpi': 100,
    }
)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / (name + '.png'), dpi=220, facecolor='white', bbox_inches='tight')
    fig.savefig(FIG / (name + '.svg'), facecolor='white', bbox_inches='tight')
    plt.close(fig)
    return FIG / (name + '.svg')


def comma(v, nd=2):
    return ('%.*f' % (nd, v)).replace('.', ',')


# ------------------------------------------------------------------ данные
def lengths(df, name='eda_lengths'):
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    n_words = df['text'].str.split().str.len()
    bins = np.arange(0, n_words.max() + 2) - 0.5
    for lab, col in ((0, C1), (1, C2)):
        ax.hist(
            n_words[df['toxic'] == lab],
            bins=bins,
            histtype='step',
            color=col,
            linewidth=1.6,
            label='%s (медиана %d)' % (CLASS_NAMES[lab], n_words[df['toxic'] == lab].median()),
        )
    ax.set_xlabel('Длина комментария, слов')
    ax.set_ylabel('Число комментариев')
    ax.legend()
    return save(fig, name)


def top_tokens(rows, name='eda_tokens'):
    """rows: [(токен, логарифм отношения шансов, частота)], первая половина - нетоксичные."""
    fig, ax = plt.subplots(figsize=(6.5, 3.9))
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    cols = [C2 if v > 0 else C1 for v in vals]
    y = np.arange(len(rows))
    ax.barh(y, vals, color=cols, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_xlabel('Логарифм отношения шансов: < 0 - признак нетоксичного, > 0 - токсичного')
    ax.grid(axis='y', visible=False)
    return save(fig, name)


def coverage(rows, name='eda_coverage'):
    """rows: [(подпись, доля неизвестных токенов проверочной выборки)]."""
    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] * 100 for r in rows], color=C1, height=0.55)
    for i, r in enumerate(rows):
        ax.text(
            r[1] * 100 + 0.4, i, comma(r[1] * 100, 1) + ' %', va='center', color=INK, fontsize=10
        )
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel('Доля токенов, неизвестных модели, %')
    ax.set_xlim(0, max(r[1] for r in rows) * 100 * 1.18 + 1)
    ax.grid(axis='y', visible=False)
    return save(fig, name)


# ------------------------------------------------------------------ эксперименты
def bars(
    summary,
    name,
    xlabel='Точность на проверочной выборке',
    target=None,
    xlim=None,
    figsize=(6.5, None),
):
    """Горизонтальные столбцы: средняя точность по трём запускам
    и разброс (стандартное отклонение)."""
    h = figsize[1] or 0.42 * len(summary) + 0.9
    fig, ax = plt.subplots(figsize=(figsize[0], h))
    y = np.arange(len(summary))
    acc = np.array([s['val_acc'] for s in summary])
    std = np.array([s['val_acc_std'] for s in summary])
    best = int(acc.argmax())
    cols = [C2 if i == best else C1 for i in range(len(summary))]
    ax.barh(
        y,
        acc,
        xerr=std,
        color=cols,
        height=0.6,
        error_kw={'ecolor': INK2, 'elinewidth': 1, 'capsize': 3},
    )
    lo = xlim[0] if xlim else max(0.0, acc.min() - 0.08)
    hi = xlim[1] if xlim else min(1.0, acc.max() + 0.06)
    for i, (a, s) in enumerate(zip(acc, std)):
        ax.text(
            min(a + s + 0.004, hi - 0.001),
            i,
            comma(a, 3),
            va='center',
            ha='left' if a + s + 0.03 < hi else 'right',
            color=INK,
            fontsize=10,
        )
    if target:
        ax.axvline(target, color=INK2, linestyle='--', linewidth=1)
        ax.text(
            target,
            len(summary) - 0.35,
            ' цель %s' % comma(target, 2),
            color=INK2,
            fontsize=9,
            va='bottom',
        )
    ax.set_yticks(y)
    ax.set_yticklabels([s['name'] for s in summary])
    ax.invert_yaxis()
    ax.set_xlim(lo, hi)
    ax.set_xlabel(xlabel)
    ax.grid(axis='y', visible=False)
    return save(fig, name)


def group_grid(groups, name='exp_groups'):
    """Пять серий подбора характеристик на одном рисунке (общая шкала точности), сетка 3 x 2."""
    titles = {
        'kernels': 'а) ширина окон свёрток',
        'filters': 'б) число фильтров в ветви',
        'layers': 'в) слои и обходные соединения',
        'activation': 'г) функция активации',
        'hidden': 'д) скрытые полносвязные слои',
    }
    keys = [k for k in titles if k in groups]
    allacc = [s['val_acc'] for k in keys for s in groups[k]]
    lo, hi = min(allacc) - 0.03, max(allacc) + 0.02
    fig, axes = plt.subplots(3, 2, figsize=(6.5, 5.2))
    axes = axes.ravel()
    for ax, k in zip(axes, keys):
        rows = groups[k]
        acc = np.array([s['val_acc'] for s in rows])
        std = np.array([s['val_acc_std'] for s in rows])
        best = int(acc.argmax())
        y = np.arange(len(rows))
        ax.barh(y, acc - lo, left=lo, xerr=std, color=[C2 if i == best else C1 for i in range(len(rows))],
                height=0.62, error_kw={'ecolor': INK2, 'elinewidth': 1, 'capsize': 2})
        for i, a in enumerate(acc):
            ax.text(lo + 0.002, i, comma(a, 3), va='center', ha='left', color='white', fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels([s['name'] for s in rows], fontsize=8)
        ax.set_ylim(4.6, -0.6)
        ax.set_xlim(lo, hi)
        ax.tick_params(axis='x', labelsize=8)
        ax.set_title(titles[k], loc='left', fontsize=9.5, color=INK)
        ax.grid(axis='y', visible=False)
    for ax in axes[len(keys):]:
        ax.axis('off')
    fig.text(0.5, 0.01, 'Средняя точность на проверочной выборке (3 запуска)', ha='center', fontsize=9.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return save(fig, name)


def timing(rows, fit, name='exp_timing'):
    """Время эпохи в зависимости от вычислительной сложности сети; линия - линейная регрессия."""
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    x = np.array([r['conv_macs'] for r in rows]) / 1000
    t = np.array([r['epoch_time'] for r in rows])
    ax.scatter(x, t, s=36, color=C1, edgecolor='white', linewidth=1.5, zorder=3)
    xs = np.linspace(0, x.max() * 1.05, 50)
    ax.plot(
        xs, fit[0] + fit[1] * xs * 1000, color=C2, linewidth=1.5, label='аппроксимация t = a + b·M'
    )
    for r, xi, ti in zip(rows, x, t):
        if r['name'] in ('F=256', '3 слоя, F=128', '2 слоя, F=128', 'F=128'):
            ax.annotate(
                r['name'],
                (xi, ti),
                textcoords='offset points',
                xytext=(6, -4),
                fontsize=9,
                color=INK2,
            )
    ax.set_xlabel('M - умножений со сложением на одну позицию текста, тыс.')
    ax.set_ylabel('Время эпохи, с')
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.legend(loc='upper left')
    return save(fig, name)


# ------------------------------------------------------------------ обучение
def curves(hist, name, title_note=''):
    """Потери и точность по эпохам: обучающая и проверочная выборки, отмечена лучшая эпоха."""
    ep = [r['epoch'] for r in hist]
    best = min(hist, key=lambda r: r['val_loss'])['epoch']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 2.6))
    a1.plot(ep, [r['train_loss'] for r in hist], color=C1, label='обучающая')
    a1.plot(ep, [r['val_loss'] for r in hist], color=C2, label='проверочная')
    a2.plot(ep, [r['train_acc'] for r in hist], color=C1, label='обучающая')
    a2.plot(ep, [r['val_acc'] for r in hist], color=C2, label='проверочная')
    for a in (a1, a2):
        a.axvline(best, color=INK2, linestyle=':', linewidth=1)
        a.set_xlabel('Эпоха')
        step = 1 if len(ep) <= 12 else (2 if len(ep) <= 20 else 5)
        a.set_xticks([e for e in ep if e == 1 or e % step == 0])
    a1.set_ylabel('Потери (перекрёстная энтропия)')
    a2.set_ylabel('Точность (accuracy)')
    a2.axhline(0.7, color=INK2, linestyle='--', linewidth=0.8)
    a1.legend(fontsize=9)
    return save(fig, name)


def lr_curves(hists, name='imp_lr'):
    """Потери на проверочной выборке по эпохам для разных скоростей обучения."""
    fig, ax = plt.subplots(figsize=(6.5, 2.5))
    for (label, hist), col in zip(hists, (C1, C2, C3)):
        ax.plot([r['epoch'] for r in hist], [r['val_loss'] for r in hist], color=col, label=label)
    ax.set_xlabel('Эпоха')
    ax.set_ylabel('Потери на проверочной выборке')
    ax.legend()
    return save(fig, name)


def confusion(cm, name='eval_confusion'):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.imshow(cm, cmap='Blues', vmin=0, vmax=cm.sum(1).max())
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                '%d\n(%s %%)' % (cm[i, j], comma(100 * cm[i, j] / cm[i].sum(), 1)),
                ha='center',
                va='center',
                color='white' if cm[i, j] > cm.max() / 2 else INK,
                fontsize=10,
            )
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel('Предсказанный класс')
    ax.set_ylabel('Истинный класс')
    ax.grid(False)
    return save(fig, name)


def roc_pr(series, name='eval_roc'):
    """series: [(подпись, вероятности, метки)] - ROC-кривые
    и кривые «точность - полнота» нескольких моделей."""
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3.1))
    for (label, probs, labels), col in zip(series, (C1, C2, C3)):
        fpr, tpr, _ = roc_curve(labels, probs)
        prec, rec, _ = precision_recall_curve(labels, probs)
        a1.plot(
            fpr,
            tpr,
            color=col,
            linewidth=1.6,
            label='%s (AUC %s)' % (label, comma(auc(fpr, tpr), 3)),
        )
        a2.plot(rec, prec, color=col, linewidth=1.6)
    a1.plot([0, 1], [0, 1], color=INK2, linestyle='--', linewidth=0.8)
    a1.set_xlabel('Доля ложных срабатываний')
    a1.set_ylabel('Полнота')
    a2.set_xlabel('Полнота')
    a2.set_ylabel('Точность срабатываний')
    a2.set_ylim(0.45, 1.01)
    handles, labs = a1.get_legend_handles_labels()
    fig.legend(handles, labs, loc='lower center', ncol=1, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.savefig(FIG / (name + '.png'), dpi=220, facecolor='white', bbox_inches='tight')
    fig.savefig(FIG / (name + '.svg'), facecolor='white', bbox_inches='tight')
    plt.close(fig)
    return FIG / (name + '.svg')


def threshold(labels, probs, name='eval_threshold'):
    from sklearn.metrics import f1_score, precision_score, recall_score

    ts = np.linspace(0.05, 0.95, 37)
    p = [precision_score(labels, probs >= t, zero_division=1) for t in ts]
    r = [recall_score(labels, probs >= t) for t in ts]
    f = [f1_score(labels, probs >= t) for t in ts]
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    ax.plot(ts, p, color=C1, label='точность срабатываний (precision)')
    ax.plot(ts, r, color=C2, label='полнота (recall)')
    ax.plot(ts, f, color=C3, label='F1-мера')
    ax.axvline(0.5, color=INK2, linestyle=':', linewidth=1)
    ax.set_xlabel('Порог вероятности, начиная с которого комментарий считается токсичным')
    ax.set_ylabel('Значение метрики')
    ax.legend(loc='lower center', fontsize=9, ncol=3)
    ax.set_ylim(0.4, 1.02)
    return save(fig, name)


def prob_hist(labels, probs, name='eval_probs'):
    fig, ax = plt.subplots(figsize=(6.5, 2.7))
    bins = np.linspace(0, 1, 21)
    for lab, col in ((0, C1), (1, C2)):
        ax.hist(
            probs[labels == lab],
            bins=bins,
            histtype='step',
            linewidth=1.6,
            color=col,
            label='истинный класс: ' + CLASS_NAMES[lab],
        )
    ax.set_xlabel('Вероятность класса «токсичный», выданная сетью')
    ax.set_ylabel('Число комментариев')
    ax.legend(loc='upper center')
    return save(fig, name)


def robustness(
    rows,
    name='eval_robust',
    xlabel='Точность на тестовой выборке',
    labels=('исходные комментарии', 'искажённые комментарии'),
    xlim=(0.5, 1.0),
):
    """rows: [(модель, точность на исходной тестовой, точность на искажённой)]."""
    fig, ax = plt.subplots(figsize=(6.5, 0.42 * len(rows) + 1.1))
    y = np.arange(len(rows))
    hgt = 0.36
    ax.barh(y - hgt / 2, [r[1] for r in rows], height=hgt, color=C1, label=labels[0])
    ax.barh(y + hgt / 2, [r[2] for r in rows], height=hgt, color=C2, label=labels[1])
    for i, r in enumerate(rows):
        ax.text(r[1] + 0.005, i - hgt / 2, comma(r[1], 3), va='center', fontsize=9, color=INK)
        ax.text(r[2] + 0.005, i + hgt / 2, comma(r[2], 3), va='center', fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9)
    ax.grid(axis='y', visible=False)
    return save(fig, name)

# -*- coding: utf-8 -*-
"""Обучение и оценка: цикл эпох, функция потерь, оптимизатор,
ранняя остановка, метрики качества."""

import copy
import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn


@torch.no_grad()
def predict_proba(model, loader):
    """Вероятность класса «токсичный» для каждого примера загрузчика и истинные метки."""
    model.eval()
    probs, labels, loss_sum = [], [], 0.0
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    for x, g, y in loader:
        logits = model(x, g)
        loss_sum += loss_fn(logits, y).item()
        probs.append(torch.softmax(logits, dim=1)[:, 1])
        labels.append(y)
    probs = torch.cat(probs).numpy()
    labels = torch.cat(labels).numpy()
    return probs, labels, loss_sum / len(labels)


def metrics(labels, probs, threshold=0.5):
    pred = (probs >= threshold).astype(int)
    return {
        'accuracy': accuracy_score(labels, pred),
        'precision': precision_score(labels, pred, zero_division=0),
        'recall': recall_score(labels, pred, zero_division=0),
        'f1': f1_score(labels, pred, zero_division=0),
        'roc_auc': roc_auc_score(labels, probs),
        'confusion': confusion_matrix(labels, pred).tolist(),
    }


# fmt: off
def train(model, train_loader, val_loader, epochs=20, lr=1e-3,
          weight_decay=0.0, patience=4, scheduler=False,
          verbose=False):
    """Обучает модель методом Adam с функцией потерь
    «перекрёстная энтропия».

    После каждой эпохи считает потери и точность на обучающей
    и проверочной выборках. Ранняя остановка: если потери на
    проверочной выборке не уменьшаются patience эпох подряд,
    обучение прекращается, и модель возвращается к весам
    лучшей эпохи.
    """
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    sched = None
    if scheduler:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.5, patience=1)
    history = []
    best, best_loss, bad = None, float('inf'), 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        tot, correct, loss_sum = 0, 0, 0.0
        for x, g, y in train_loader:
            opt.zero_grad()
            logits = model(x, g)          # прямой проход
            loss = loss_fn(logits, y)     # потери
            loss.backward()               # обратный проход
            opt.step()                    # шаг Adam
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            tot += len(y)
        epoch_time = time.perf_counter() - t0
        probs, labels, val_loss = predict_proba(model,
                                                val_loader)
        row = {
            'epoch': epoch,
            'train_loss': loss_sum / tot,
            'train_acc': correct / tot,
            'val_loss': val_loss,
            'val_acc': float(((probs >= 0.5) == labels).mean()),
            'time': epoch_time,
            'lr': opt.param_groups[0]['lr'],
        }
        history.append(row)
        if verbose:
            print('эпоха {epoch:2d}: потери {train_loss:.4f} / '
                  '{val_loss:.4f}, точность {train_acc:.3f} / '
                  '{val_acc:.3f}, {time:.2f} с'.format(**row))
        if sched is not None:
            sched.step(val_loss)
        if val_loss < best_loss - 1e-4:   # потери уменьшились
            best_loss, bad = val_loss, 0
            best = copy.deepcopy(model.state_dict())
            row['best'] = True
        else:                             # эпоха без улучшения
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best)
    return history
# fmt: on


def summarize(history):
    best = min(history, key=lambda r: r['val_loss'])
    return {
        'epochs_run': len(history),
        'best_epoch': best['epoch'],
        'val_loss': best['val_loss'],
        'val_acc': best['val_acc'],
        'epoch_time': float(np.mean([r['time'] for r in history])),
    }

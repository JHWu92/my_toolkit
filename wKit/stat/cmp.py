# coding=utf-8
import pandas as pd
import numpy as np


def cmp_rank_list(base, new, gold=None):
    """

    Parameters
    ----------
    base and  new: compare new to base.
        1. array-like: items only, whose order is the rank.
        2. 2darray-like: shape=(N, 2), N samples. (hashable, float/int).
            [:, 0] are items, [:, 1] are scores on which the rank is based
    gold: list, gold standard rank.

    :return:
    columns: ['item', 'rank_change', 'score_change', 'rank_new', 'rank_base', 'score_new', 'score_base']
        or   ['item', 'rank_change', 'rank_new', 'rank_base'] if no score is provided
    """
    base = np.array(base)
    new = np.array(new)
    assert len(base.shape) == len(new.shape), 'number of columns should be the same'
    assert len(base.shape) in {1, 2}, 'only 1 or 2 columns are allowed'
    has_score = True if len(base.shape) == 2  else False
    base, base_scores = (base[:, 0], base[:, 1].astype(float)) if has_score else (base, None)
    new, new_scores = (new[:, 0], new[:, 1].astype(float)) if has_score else (new, None)

    base_items = set(base)
    new_items = set(new)
    assert len(base_items) == len(base) and len(new_items) == len(new), 'Duplicates in list'

    def add_gold_rank(temp, item, rank=None):
        if gold is not None and item in gold:
            gold_rank = gold.index(item)
            temp['gold_rank'] = gold_rank
            if rank is not None:
                change = gold_rank-rank
                temp['gold_change'] = '+%d' % change if change > 0 else str(change)
                # temp['gold_change'] = change
    res = []
    for rank, item in enumerate(new):
        if item in base_items:
            rank_base = np.where(base == item)[0][0]
            change = rank_base - rank
            change = '+%d' % change if change > 0 else str(change)
            tmp = {'item': item, 'rank_change': change, 'rank_new': rank, 'rank_base': rank_base}
            add_gold_rank(tmp, item, rank)
            if has_score:
                ns, bs = new_scores[rank], base_scores[rank_base]
                score_change = ns * 100.0 / bs - 100
                # score_change = '+{:.02f}%'.format(score_change) if score_change > 0 else '{:.02f}%'.format(score_change)
                tmp.update({'score_new': ns, 'score_base': bs, 'score_change%': score_change})
            res.append(tmp)
        else:
            tmp = {'item': item, 'rank_change': '+', 'rank_new': rank}
            add_gold_rank(tmp, item, rank)
            if has_score:
                tmp['score_new'] = new_scores[rank]
            res.append(tmp)

    for rank, item in enumerate(base):
        if item not in new_items:
            tmp = {'item': item, 'rank_change': '-', 'rank_base': rank}
            add_gold_rank(tmp, item)
            if has_score:
                tmp['score_base'] = base_scores[rank]
            res.append(tmp)

    df = pd.DataFrame(res)
    cols = ['item']
    if gold is not None: cols += ['gold_rank', 'gold_change']
    cols += ['rank_change']
    if has_score: cols += ['score_change%']
    cols += ['rank_new', 'rank_base']
    if has_score: cols += ['score_new', 'score_base']
    for col in list(set(cols) - set(df.columns)):
        df[col] = np.nan
    df.fillna('', inplace=True)
    return df[cols]

# coding=utf-8
import pandas as pd


def group_consecutive(data, stepsize=1):
    """
    group consecutive number as as sub list.
    E.g. data = [1, 2, 3, 5, 6, 7, 10, 11]
    stepsize=1: return [[1,2,3], [5,6,7], [10,11]]
            =2: return [[1,2,3,5,6,7], [10,11]]
    :param data: list/array
    :param stepsize: define consecutive.
    """
    import numpy as np
    return np.split(data, np.where(np.diff(data) > stepsize)[0] + 1)


def group_consecutive_duplicates(data, with_cnt=True):
    """
    yield elements after consecutive duplicates grouping.
    E.g.
    >>> data= [0, 0, 0, 1, 2, 3, 3, 3]
    >>> list(group_consecutive_duplicates(data))
    [(0, 3), (1, 1), (2, 1), (3, 3)]

    >>> list(group_consecutive_duplicates(data, with_cnt=False))
    [0, 1, 2, 3]
    """
    from itertools import groupby
    for k, g in groupby(data):
        if with_cnt:
            yield (k, len(list(g)))
        else:
            yield k


def even_chunks(array, max_chunk_size, indices=False, right_close=False):
    """Yield chunks with (almost) even size.
    For example:
    >>> arr = [1,2,3,4,5,6,7]

    >>> list(get_chunks(arr, 5))
    [[1, 2, 3, 4, 5], [6, 7]]

    >>> list(even_chunks(arr, 5))
    [[1, 2, 3, 4], [5, 6, 7]]

    """
    import math
    size = len(array)
    num_chunks = math.ceil(size * 1.0 / max_chunk_size)
    new_chunk_size = int(math.ceil(size * 1.0 / num_chunks))
    return get_chunks(array, new_chunk_size, indices, right_close=right_close)


def get_chunks(array, chunk_size, indices=False, right_close=False):
    """Yield successive chunks with chunk_size from array.
    params:
        indices: if false, yield chunks of array; if True, yield indices pair (left, right) only
        right_close: if False return elements with indices in [left, right); if True, return indices in [left, right]

    For example,

    >>> arr = [1,2,3,4,5,6,7]
    >>> list(get_chunks(arr, 5))
    [[1, 2, 3, 4, 5], [6, 7]]

    >>> list(get_chunks(arr, 5, right_close=True))
    [[1, 2, 3, 4, 5, 6], [6, 7]]

    >>> idx = list(get_chunks(arr, 5, indices=True))
    >>> idx
    [(0, 5), (5, 7)]

    >>> arr[idx[0][0]:idx[0][1]], arr[idx[1][0]: idx[1][1]]
    ([1, 2, 3, 4, 5], [6, 7])

    """

    for i in range(0, len(array), chunk_size):
        left = i
        right = min(len(array), i + chunk_size + right_close)
        if indices:
            yield (left, right)
        else:
            yield array[left: right]


def downsample_by_step(l, step):
    return l[::step]


def downsample_by_step_include_last(l, step=5):
    down_l = downsample_by_step(l, step)
    if len(l) % step != 1:
        down_l.append(l[-1])
    return down_l


def sort_by_group_sum(df):
    ":var df: pandas.DataFrame"
    grp = df.groupby('place')
    sort1 = df.ix[grp[['gt_visit']].transform(sum).sort('gt_visit', ascending=False).index]
    f = lambda x: x.sort('ym')
    sort2 = sort1.groupby('place', sort=False).apply(f)
    sort2 = sort2.reset_index(0, drop=True)
    return sort2


def find_tree(messy_tree_df, parent_node='root', lv=1):
    """
    the nodes which become parent-less
    after we remove the top level parent,
    are the children of the removed parent
    """
    new_tree = []
    all_nodes = set(messy_tree_df.child)
    nodes_have_true_parent = set(messy_tree_df[messy_tree_df.child != messy_tree_df.parent].child)
    nodes_without_parent = all_nodes - nodes_have_true_parent
    for child in nodes_without_parent:
        new_tree.append((child, parent_node))
    remaining_nodes = messy_tree_df[~messy_tree_df.child.isin(nodes_without_parent)]
    for node in list(nodes_without_parent):
        # remove the inode
        sub_tree = find_tree(remaining_nodes[remaining_nodes.parent != node], node, lv + 1)
        new_tree.extend(sub_tree)
    if lv != 1:
        return new_tree
    return pd.DataFrame(new_tree, columns=['child', 'parent'])
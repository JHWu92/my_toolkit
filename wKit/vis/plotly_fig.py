# coding=utf-8
from plotly import graph_objs as go


def box(arrs, names=None, title=None, orientation='v', reverse=False):
    """
    boxplot with plotly

    Parameters
    ----------
    arrs: list of arr. Each arr is one box.
    names: list. Names for each box.
    title: str. title of the graph. Default None
    orientation: {'v', 'h'}, default v(verticle)
    reverse: boolean. Default False.
        If True:  'v': up->down; 'h': right->left
        If False: 'v': down->up; 'h': left->right

    Return
    ----------
    plotly.graph_objs.Figure for iplot/plot
    """

    size = len(arrs)
    if names is None:
        names = ['box %d' % i for i in range(size)]
    assert len(names) == size, 'len of arrs dosent match with len of names'

    order = range(size) if not reverse else range(size - 1, -1, -1)
    if orientation == 'h':
        data = [go.Box(x=arrs[i], name=names[i]) for i in order]
    elif orientation == 'v':
        data = [go.Box(y=arrs[i], name=names[i]) for i in order]
    else:
        raise ValueError("orienation should be in {'v', 'h'}")

    layout = go.Layout(title=title)
    fig = go.Figure(data=data, layout=layout)
    return fig


def hist(arrs, names=None, title=None, orientation='v'):
    """
    hist with plotly

    Parameters
    ----------
    arrs: list of arr. Each arr is one hist.
    names: list. Names for each hist.
    title: str. title of the graph. Default None
    orientation: {'v', 'h'}, default v(verticle)

    Return
    ----------
    plotly.graph_objs.Figure for iplot/plot
    """

    size = len(arrs)
    if names is None:
        names = ['hist %d' % i for i in range(size)]
    assert len(names) == size, 'len of arrs dosent match with len of names'

    order = range(size)
    if orientation == 'v':
        data = [go.Histogram(x=arrs[i], name=names[i]) for i in order]
    elif orientation == 'h':
        data = [go.Histogram(y=arrs[i], name=names[i]) for i in order]
    else:
        raise ValueError("orienation should be in {'v', 'h'}")

    layout = go.Layout(title=title)
    fig = go.Figure(data=data, layout=layout)
    return fig
def str2date_hist(df, col, form='%m/%d/%Y'):
    try:
        time = pd.to_datetime(df[col], format=form)
    except ValueError:
        time = pd.to_datetime(df[col])
    return time.hist()

def str2date_range(df, col,  form='%m/%d/%Y', to_str=False):
    try:
        time = pd.to_datetime(df[col], format=form)
    except ValueError:
        time = pd.to_datetime(df[col])
    if to_str:
        return '%s - %s' % (time.min().strftime('%m/%d/%Y'), time.max().strftime('%m/%d/%Y'))
    return time.min(), time.max()
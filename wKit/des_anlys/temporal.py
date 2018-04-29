def float_char_time_range(names, start_dates, end_dates, ticks_by_year=False):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # y values
    yval = list(range(len(names)))
    # x values
    start = mdates.date2num(start_dates)
    end = mdates.date2num(end_dates)
    width = end-start
    
    fig, ax = plt.subplots()
    ax.barh(bottom=yval, width=width, left=start, height=0.3)
    xfmt = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(xfmt)
    ax.xaxis.grid(linestyle='dashed')
    ax.set_axisbelow(True)  # grid lines under bar
    ax.yaxis.set_ticks(yval)
    ax.yaxis.set_ticklabels(names)
    # autorotate the dates
    fig.autofmt_xdate()
    w = 10
    if ticks_by_year:
        min_year = min(start_dates).year
        max_year = max(end_dates).year
        ticks = mdates.date2num([mdates.datetime.datetime(y, 1,1) for y in range(min_year, max_year+1)] + [max(end_dates)])
        ax.xaxis.set_ticks(ticks)
        w = (len(ticks)-1)*0.7
        
    fig.set_size_inches(w=w, h=0.7*len(yval))
    plt.show()
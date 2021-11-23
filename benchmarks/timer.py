import time


timestamps = dict()
timedata = dict()


def clear():
    timestamps.clear()
    timedata.clear()


def start(name):
    timestamps[name] = time.time()


def stop(name):
    t = time.time() - timestamps[name]
    if name not in timedata:
        timedata[name] = []
    timedata[name].append(t)


def get(name, window=10):
    if name not in timedata:
        raise RuntimeError('No data found for {}'.format(name))
    d = timedata[name][-window:]
    if len(d) == 0:
        raise RuntimeError('Empty data found for {}'.format(name))
    variance = len(d) * sum([x ** 2 for x in d]) - sum(d) ** 2
    mean = sum(d) / len(d)
    return mean, variance


def report(names, window=10):
    ts = []
    for name in names:
        t, v = get(name, window=window)
        ts.append('{}: {:.5f} ({:.5f}) ms'.format(name, t * 1e3, v * 1e3))
    return ' | '.join(ts)

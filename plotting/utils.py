def processlabel(label):
    if label.startswith('ds'):
        return 'ZeRO stage ' + label.split('-')[-1]
    if label == 'fastmoe':
        return 'No optimization'
    if label == 'dynamic' or label.startswith('dynrep'):
        return 'Shadow'
    if label == 'smart_dynamic':
        return 'Shadow + smart sch.'
    if label.startswith('smart-scheduling') or label.startswith('smart-schd'):
        return 'Smart scheduling'
    if label.startswith('top'):
        return 'Hybrid ' + label
    if label == 'so':
        return 'Hybrid with all optimizations'
    return label

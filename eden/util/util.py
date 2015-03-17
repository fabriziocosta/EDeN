
def report_base_statistics(vec):
    from collections import Counter
    c = Counter(vec)
    msg = ''
    for k in c:
        msg += "class: %s count:%d (%0.2f)\t" % (k, c[k], c[k] / float(len(vec)))
    return msg

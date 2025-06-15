def calculate_auc(y_true, y_scores):
    data = list(zip(y_scores, y_true))
    data.sort(key=lambda x: -x[0])

    np = sum(y_true) 
    nn = len(y_true) - np
    tp,tn = 0,0
    pp,pn = 0,0 # previous

    res = 0
    for i in range(len(data)):
        if data[i][1] == 1:
            tp += 1
        else:
            tn += 1
        res += (tp+pp)/np * (tn-pn)/nn * 0.5
        pp,pn = tp,tn
    return res
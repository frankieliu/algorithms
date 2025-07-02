def read_file(filename):
    h = dict()
    with open(filename) as f:
        for line in f.readlines():
            kv = re.split(r"\s+", line)
            if len(kv) != 2:
                # Raise and error here
                continue
            h[kv[0]] = kv[1]
    return h

def main():
    if len(sys.argv) != 3:
        return
    files = [arg for arg in sys.argv]
    o = read_file(files[0])
    l = read_file(files[1])
    r = read_file(files[2])
    # merge conflicts
    # 1. both change a value: ok if the same value
    # 1. one deletes and the other changes a value
    # 1. both add the same key and different value
    #
    # addition, deletion, update
    # add + add - conflict
    # add + del - does not happen
    # add + upd - does not happen
    # del + upd - conflict
    # del + del - safe 
    # upd + upd - conflict 
    o_keys = set(o.keys())
    l_keys = set(l.keys())
    r_keys = set(r.keys())

    out = o.copy()
    # intersection: update + updates
    inter_keys = o_keys & l_keys & r_keys
    for k in inter_keys:
        if l[k] != r[k]:
            out[k] = '?'

    # update and delete:
    a_keys, b_keys = l_keys, r_keys
    up_del = (o_keys & a_keys) - b_keys
    for k in up_del:
        if o_keys == a_keys:
            del out[k]
        else:
            out[k] = "?"

    a_keys, b_keys = r_keys, l_keys
    up_del = (o_keys & a_keys) - b_keys
    for k in up_del:
        if o_keys == a_keys:
            del out[k]
        else:
            out[k] = "?"

    # delete delete 
    del_del = (o_keys - r_keys) & (o_keys - l_keys)
    for k in del_del:
        del out[k]

    # add add
    add_add = (r_keys - o_keys) & (l_keys - o_keys)
    for k in add_add:
        if k in r_keys and k in l_keys:
            if r[k] == l[k]:
                out[k] = r[k]
            else:
                out[k] = "?"
    
    # add
    a_keys,b_keys = l_keys, r_keys
    a = l  
    add = (a_keys - o_keys) - (b_keys - o_keys)
    for k in add:
        out[k] = a[k]

    # add
    a_keys,b_keys = r_keys, l_keys
    a = r  
    add = (a_keys - o_keys) - (b_keys - o_keys)
    for k in add:
        out[k] = a[k]
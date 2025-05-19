import re
with open("Data/links.md") as f:
    l = f.readlines()
    for el in l:
        el = el.strip()
        m = re.split(r"\/+", el)
        if len(m) != 6:
            continue
        match = re.search(r"manager",m[5])
        if not match:
            print(m)

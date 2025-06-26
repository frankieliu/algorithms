# %%

def permute_unique(a):
    a = sorted(a)
    n = len(a)
    used = [False]*n
    res = []

    def bt(path):
        if len(path) == n:
            res.append(path.copy())
        for i in range(n):
            if used[i] or (i > 0 and a[i] == a[i-1] and not used[i-1]):
                continue
            used[i] = True
            path.append(a[i])
            bt(path)
            path.pop()
            used[i] = False
    bt([])
    return res

# %%


def main():
    print(permute_unique([1, 2, 2, 3]))


if __name__ == "__main__":
    main()

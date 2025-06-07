def z(s):
    l, r = -1, -1
    z = [0] * len(s)
    for i in range(1, len(s)):
        j = i
        if j < r:
            j = i + min(z[j - l], r - i)
        while j < len(s) and s[j] == s[j - i]:
            j += 1
        if j > r:
            l = i
            r = j
        z[i] = j - i
    return z


if __name__ == "__main__":
    print(z("aaaaaa"))

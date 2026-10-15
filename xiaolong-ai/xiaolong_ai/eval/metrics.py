# -*- coding: utf-8 -*-

# 小龙智脑 (XiaoLong Brain) - 全新原创项目

# 作者 / 版权人: 小龙 (XiaoLong)

# License: MIT。本项目所有代码均为原创，保留署名即可自由使用。



import re





def normalize(text):

    return " ".join(re.findall(r"[a-z0-9一-鿿]+", (text or "").lower()))





def exact_match(pred, gold):

    return normalize(pred) == normalize(gold)





def token_f1(pred, gold):

    p = normalize(pred).split()

    g = normalize(gold).split()

    if not p and not g:

        return 1.0

    common = set(p) & set(g)

    if not common:

        return 0.0

    prec = len(common) / len(p)

    rec = len(common) / len(g)

    return 2 * prec * rec / (prec + rec)





def rouge_l(pred, gold):

    p = normalize(pred).split()

    g = normalize(gold).split()

    if not g:

        return 1.0 if not p else 0.0

    m, n = len(p), len(g)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):

        for j in range(1, n + 1):

            if p[i - 1] == g[j - 1]:

                dp[i][j] = dp[i - 1][j - 1] + 1

            else:

                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]

    rec = lcs / len(g)

    prec = lcs / len(p) if p else 0.0

    if prec + rec == 0:

        return 0.0

    return 2 * prec * rec / (prec + rec)





def evaluate(pred, gold):

    return {

        "exact_match": exact_match(pred, gold),

        "token_f1": token_f1(pred, gold),

        "rouge_l": rouge_l(pred, gold),

    }


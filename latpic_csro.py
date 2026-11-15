# -*- coding: utf-8 -*-
"""在 latpic 照片上算 Warren-Cowley 短程序参数（CSRO）。

这里只做一件事：把一个"点阵上的占位"变成"每种元素对在每个近邻壳上有多少对"，
再按 Warren-Cowley 的定义除下去。核心观察是——

    在固定点阵上，第 L 壳的邻居关系与格点上**坐着谁**无关，
    它只是格点之间的一个置换 / 邻接矩阵 S_L。

所以配对计数就是一次矩阵乘：

    n_L = Xᵀ · S_L · X          X = 每行一个格点的 one-hot,形状 (M, S)

S_L 只跟模板有关、与结构无关 —— 建一次，所有结构复用。于是单个结构的成本
从"每原子查一次 KD 树 + 每行一次 pandas"降成"几次 gather + 一个 (M,S)ᵀ·(M,S)"。

Warren-Cowley 定义（与 LEP 的 `lep/tools/csro.py` 完全一致）：

    α_ij = 1 − n_ij / (M · Z · c_i · c_j)

    n_ij  = 有序对数（中心是 i，邻居是 j）的**多重度加权**计数
    M     = 参与的格点数，Z = 参与统计的壳里每格点的邻居数（含多重度）
    c_i   = i 的摩尔分数

一条判读规则就够：**α_ij < 0 表示 i–j 对出现的次数高于随机**（i 和 j 互相吸引），
α_ij > 0 表示低于随机（互相回避）。落到具体情形是：

    α_ij < 0, i ≠ j  → 异类结伴 = 有序（ordering）
    α_ij > 0, i ≠ j  → 异类回避 = 偏聚（clustering）
    α_ii < 0         → 同类偏聚
    α_ii > 0         → 分散均匀

注意对角与非对角的符号方向**相反**：B2 有序结构的 α_AlTi = −1、α_AlAl = +1。
"""

import numpy as np

__all__ = ["CSROEngine", "alpha_from_counts", "csro_report"]


def alpha_from_counts(counts, conc, n_sites, n_neighbor):
    """α_ij = 1 − n_ij / (M·Z·c_i·c_j)；c 为 0 的对保持 nan。"""
    denom = n_sites * n_neighbor * np.outer(conc, conc)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 - counts / denom


class CSROEngine:
    """把模板的近邻关系预先算好，之后每个结构只剩一次矩阵乘。

    Parameters
    ----------
    template : LatticeTemplate
    n_shells : int
        要算到第几近邻壳。**这是模板一次性的开销**，多算几壳几乎不涨钱。
    with_vacancy : bool
        照片里 code == 0 的格点算不算一个"元素"。默认算（记作 Va）：空位同样
        参与短程序。全占据的结构里它不会出现，白占一列而已。
    """

    def __init__(self, template, n_shells=2, with_vacancy=True):
        self.template = template
        self.n_shells = int(n_shells)
        self.with_vacancy = with_vacancy
        self.shells = []                     # [(idx (M,Z) int32, mult (M,Z) float, Z 均值)]
        for L in range(1, self.n_shells + 1):
            idx, mult = template.shell(L, with_multiplicity=True)
            mult = np.asarray(mult, dtype=np.float64)
            self.shells.append((np.ascontiguousarray(idx, np.int32), mult,
                                float(mult.sum() / mult.shape[0])))

    # ------------------------------------------------------------------ 核心
    def _labels(self, pic, cats):
        lab = []
        for c in cats:
            lab.append("Va" if c == 0 else (
                pic.species[c - 1] if c - 1 < len(pic.species) else "Z{}".format(c)))
        return lab

    def counts(self, pic):
        """返回 (labels, counts, M, conc)。

        counts[L] 是第 L 壳的 (S,S) **有序**配对计数；conc 是按同一批格点算的摩尔分数。
        """
        if pic.template.grid != self.template.grid:
            raise ValueError("照片与引擎的模板不是同一个")
        codes = pic.values()                         # (M,) 编码，0 = 空位
        if not self.with_vacancy:
            codes = codes[codes != 0]
        cats = np.unique(codes)
        M, K = len(codes), len(cats)
        X = np.zeros((M, K), np.float64)
        X[np.arange(M), np.searchsorted(cats, codes)] = 1.0
        out = []
        for idx, mult, _Z in self.shells:
            # X[idx] 是 (M, Z, K)；乘多重度再对 Z 求和 = S_L·X
            Y = np.einsum("mzk,mz->mk", X[idx], mult, optimize=True)
            out.append(X.T @ Y)
        return self._labels(pic, cats), out, M, X.mean(0)

    # ------------------------------------------------------------------ α
    def alpha(self, pic, n_shells=None, cumulative=True):
        """{(a, b): α}，键按元素序号排好，与 LEP 的 `calculate_csro` 对齐。

        cumulative=True  —— 复刻 LEP 的 `layers=L` 口径：把 1..L 壳**合并**统计；
        cumulative=False —— 每个壳单独一个 dict（LEP 那边要循环调用才能拿到）。
        """
        L = self.n_shells if n_shells is None else int(n_shells)
        if L > self.n_shells:
            raise ValueError("引擎只准备了 {} 壳".format(self.n_shells))
        lab, cnt, M, conc = self.counts(pic)
        if cumulative:
            Z = sum(self.shells[k][2] for k in range(L))
            return _as_dict(lab, alpha_from_counts(sum(cnt[:L]), conc, M, Z))
        return [_as_dict(lab, alpha_from_counts(cnt[k], conc, M, self.shells[k][2]))
                for k in range(L)]


def _as_dict(lab, A):
    """(K,K) α 矩阵 -> {(a,b): v}，只留 a<=b。"""
    return {(lab[i], lab[j]): float(A[i, j])
            for i in range(len(lab)) for j in range(len(lab)) if i <= j}


def csro_report(alpha, tol=1e-3):
    """把 α 打成一张表，并按符号给出偏聚/有序的判读。"""
    pairs = sorted(alpha, key=lambda k: (k[0] != k[1], k))
    out = ["  {:<10s}{:>10s}   判读".format("元素对", "α")]
    for a, b in pairs:
        v = alpha[(a, b)]
        if not np.isfinite(v):
            verdict = "（该元素不存在）"
        elif a != b:
            # 非对角：α<0 = 这对出现得比随机多 = 异类结伴（有序）
            verdict = "异类结伴" if v < -tol else ("异类回避" if v > tol else "近随机")
        else:
            # 对角：α<0 = 同类聚到一起
            verdict = "同类偏聚" if v < -tol else ("分散均匀" if v > tol else "近随机")
        out.append("  {:<10s}{:>10.4f}   {}".format("{}–{}".format(a, b), v, verdict))
    return "\n".join(out)

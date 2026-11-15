# -*- coding: utf-8 -*-
"""FeatureEngine —— 照片到 LEP 特征矩阵，以及逐行增量重算。

列布局与 LEP 的 `StructureAnalyzer.create_atomic_environment_df` **完全相同**：

    [0_Al 0_Nb 0_Ti ... 1_Al 1_Nb ... 2_Al ... n_L_*]

也就是「先按壳、再按元素」，组 0 是原子自己的元素（one-hot），组 L 是第 L 近邻壳里
每种元素的**多重度加权**计数。所以拿这里算出来的矩阵可以直接喂给用户现有的
torch 模型，不需要改模型那一侧。
"""

import numpy as np

# 编码 0 = 空位，1..S = species 下标 + 1
VACANCY = 0


def picture_from_codes(template, codes, species):
    """(M,) 的格点编码 -> 一张稠密的 Picture。"""
    from latpic import Picture
    code = np.zeros(template.grid, np.uint8)
    code[tuple(template.site_idx.T)] = np.asarray(codes, np.uint8)
    return Picture(template, code, species)


class FeatureEngine:
    """把「晶格的近邻关系」一次性算好，之后所有结构共用。

    Parameters
    ----------
    template : LatticeTemplate
    species  : list[str]
        模型认的元素表，**顺序就是特征列的顺序**。它可以比结构里实际出现的元素多，
        这时多出来的列恒为 0 —— 与 LEP 的 `element_list` 行为一致。
    n_shells : int
        算到第几近邻壳；特征里会有 `n_shells + 1` 组（组 0 = 自己）。
    with_vacancy : bool
        给空位单独留一列（默认关）。关掉时 code 0 的格点对特征毫无贡献，
        这正是 LEP 遇到 element_list 之外符号时的行为。
    """

    def __init__(self, template, species, n_shells=5, with_vacancy=False):
        self.template = template
        self.species = list(species)
        self.n_shells = int(n_shells)
        self.with_vacancy = bool(with_vacancy)
        self.n_sites = template.n_sites

        self.K = len(self.species) + (1 if with_vacancy else 0)
        self.n_feat = (1 + self.n_shells) * self.K

        # 编码 -> 列号（不在表里的编码给 -1，即"看不见"）
        lut = np.full(256, -1, np.int16)
        for s in range(len(self.species)):
            lut[s + 1] = s
        if with_vacancy:
            lut[VACANCY] = len(self.species)
        self._lut = lut

        # 近邻：每壳 (M,Z) 的格点序号 + 多重度。模板属性，只算一次。
        self.shell_idx, self.shell_mult = [], []
        for L in range(1, self.n_shells + 1):
            idx, mult = template.shell(L, with_multiplicity=True)
            self.shell_idx.append(np.ascontiguousarray(idx, np.int32))
            self.shell_mult.append(np.ascontiguousarray(mult, np.float64))

        # 「动了哪几个格点，就要重算哪几行的特征」：自己 + 所有壳里的邻居。
        # 壳是对称的（d 在壳里 <=> −d 也在壳里），所以影响集合天然对称。
        touch = [np.arange(self.n_sites, dtype=np.int32)[:, None]]
        touch += self.shell_idx
        self.touch = np.concatenate(touch, axis=1)          # (M, 1 + Z总)

        # 把各壳的近邻拼成**一条** (M, Z总) 的轴，每列属于第 col_shell[z] 壳。
        # MC 每步的「改了 i 之后各行特征变多少」就是一次布尔比较 + 一次 reduceat，
        # 而不是在 Python 里循环「壳数」次；-1 是永远匹配不上的哑邻居。
        Ztot = sum(a.shape[1] for a in self.shell_idx)
        self.nb_idx = np.full((self.n_sites, Ztot), -1, np.int32)
        self.nb_mult = np.zeros((self.n_sites, Ztot), np.float32)
        self.shell_starts = np.zeros(self.n_shells, np.int64)
        z = 0
        for L in range(self.n_shells):
            w = self.shell_idx[L].shape[1]
            self.shell_starts[L] = z
            self.nb_idx[:, z:z + w] = self.shell_idx[L]
            self.nb_mult[:, z:z + w] = self.shell_mult[L]
            z += w

    # ------------------------------------------------------------------ 查询
    def column(self, shell, symbol):
        """(壳, 元素名) -> 列号；找不到返回 -1。与 LEP 的 column_map 同一个语义。"""
        if symbol == "Va":
            return (1 + shell) * self.K + len(self.species) if self.with_vacancy else -1
        if symbol not in self.species:
            return -1
        return shell * self.K + self.species.index(symbol)

    @property
    def columns(self):
        names = [f"{L}_{s}" for L in range(self.n_shells + 1)
                 for s in self.species + (["Va"] if self.with_vacancy else [])]
        return names

    def describe(self):
        return ("{} 个格点，{} 壳，{} 个元素 -> 特征 {} 列 = ({}+1)×{}".format(
            self.n_sites, self.n_shells, len(self.species),
            self.n_feat, self.n_shells, self.K))

    # ------------------------------------------------------------------ one-hot
    def onehot(self, pic):
        """(M, K) 的 one-hot，行 = 格点、列 = 元素。"""
        return self.onehot_from_codes(pic.values())

    def onehot_from_codes(self, codes):
        codes = np.asarray(codes)
        col = self._lut[codes]
        OH = np.zeros((len(codes), self.K), np.float64)
        ok = col >= 0
        OH[np.arange(len(codes))[ok], col[ok]] = 1.0
        return OH

    def onehot_vector(self, code):
        """单个编码 -> (K,) 的 one-hot。空位（或看不见的元素）给全 0。

        注意**不要**用 `OH[code - 1]` 代替它：编码是「元素下标 + 1」，
        而 OH 的列是为了 `with_vacancy` 才可能多一列，两者不是同一个索引。
        """
        v = np.zeros(self.K, np.float64)
        c = self._lut[int(code)]
        if c >= 0:
            v[c] = 1.0
        return v

    # ------------------------------------------------------------------ 特征
    def features(self, pic):
        """(M, n_feat) —— 整张照片。"""
        return self.features_from_oh(self.onehot(pic))

    def features_from_codes(self, codes, rows=None):
        return self.features_from_oh(self.onehot_from_codes(codes), rows)

    def features_from_oh(self, OH, rows=None):
        """**逐行**算特征：只算 rows 指定的那些行。这是 MC 每步真正调用的东西。

        成本 O(|rows|·Z·K)，与 M 无关 —— 这就是 MC 每步从「克隆整个 (M,F)」
        降到「几十行」的原因。
        """
        if rows is None:
            r, n = np.arange(self.n_sites), self.n_sites
        else:
            r = np.asarray(rows, dtype=np.int64)
            n = len(r)
        F = np.zeros((n, self.n_feat), np.float64)
        F[:, :self.K] = OH[r]                              # 组 0：自己
        for L in range(self.n_shells):
            nb = self.shell_idx[L][r]                      # (n, Z)
            Y = np.einsum("nzk,nz->nk", OH[nb], self.shell_mult[L][r], optimize=True)
            F[:, (L + 1) * self.K:(L + 2) * self.K] = Y
        return F

    # ------------------------------------------------------------------ 影响集
    def affected(self, sites):
        """改动 sites 里的格点后，哪些行的特征需要重算。"""
        return np.unique(self.touch[np.asarray(sites, np.int64)].ravel())

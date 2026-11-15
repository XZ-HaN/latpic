#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""latpic —— 把「固定晶格格点上的占位」存成一张三维照片。

想法
----
固定晶格上的结构，原子只会落在固定格点上，所以它需要记的东西**只有每个格点上
填的是什么**。把晶胞切成一张规则的三维网格，每个网格点填一个符号（0 = 空），
这就是一张「照片」。格点几何（哪些点是格点、边长多少）另存，所有结构共用一份。

判据只有一条，三种晶格共用同一个函数：
    网格点 (i,j,k) 的分数坐标 (i/nx, j/ny, k/nz) 乘上**惯用胞**，
    如果落在一个基矢位置上，它就是格点。
于是

    简单立方 sc   网格 n³     全满                    占 1/1
    面心立方 fcc  网格 (2n)³  (i+j+k) 为偶数           占 1/2
    体心立方 bcc  网格 (2n)³  i≡j≡k (mod 2)            占 1/4

最后一条正是「bcc 空出 3/4」。同样的规则换成正交六方胞，hcp 用 nx=3, nz=2 也能拍
（基矢 (0,0,0) 与 (1/3,2/3,1/2) 正好落在网格点上）。

三层结构
--------
    LatticeTemplate   组合信息：网格多大、哪些点是格点     —— 全库共用
    Picture           照片：每个格点填什么（整数编码）     —— 每个结构一张
    cell              晶胞 3×3 矩阵或一个立方边长          —— 每个结构一份

照片**不含晶胞**，这是省空间的根源：几何不重复存，每个结构只存它的符号序列。
16 原子 bcc 的照片严格按 3 bit/位打包是 **7 个字节**，而一份 POSCAR 是 1264 字节。

两个必须记住的性质
------------------
1. **照片永远是理想点阵的几何。** 弛豫出来的位置偏移不进照片（进了就不省了）。
   所以 `to_ase()` 给出的是理想几何；要还原 DFT 的弛豫坐标，得另外存位移。
   对晶格模型（CE / LEP）和 MC 来说这恰恰是对的：能量/性质是**占位**的函数，
   理想几何才是它们定义域里的那个点。
2. **非格点的网格点恒为 0**，是模板决定的，所以存的时候不上盘、读的时候按模板
   铺开。这就是「照片」和「照片 + 模板」在体积上差 4 倍的地方。
3. **尺寸不是格式的属性，是每条照片的属性。** 库里的模板按需登记，各条的编码
   首尾相接存放，所以 2 原子、16 原子、54 原子的结构可以混在一个库里。
4. **编码只占 1 字节，而且上界由元素周期表定死**：0 = 空位，1..S = 本库符号表，
   最多 119 种状态 ⇒ 打包后每格点最多 7 bit，任何体系都不会更宽。打包用的进制
   按这张照片**实际用到**的最大编码取，所以 2 元的照片 1 bit/格点、5 元的 3 bit。
   需要真正的原子序数时用 `Picture.zcodes()`。

    from latpic import LatticeTemplate, Picture
    tpl = LatticeTemplate.bcc(2)                 # 2×2×2 惯用胞 = 16 个格点
    pic = Picture.from_ase(atoms, tpl, ["Al","Nb","Ti","V","Zr"])
    atoms2 = pic.to_ase(cell_atoms.cell)         # 理想几何
    pic.code                                     # (4,4,4) 整数照片
"""
from __future__ import annotations

import itertools
import json
import math
import os
import tempfile
from typing import Iterable, Sequence

import numpy as np

# ase 只在真正要读写结构时才用到（from_ase / to_ase / zcodes），
# 格式内核（模板、照片、打包）不该因为它没装就跑不起来。
try:
    from ase import Atoms
    from ase.build import bulk
except ImportError:                                   # pragma: no cover
    Atoms = None
    bulk = None


VACANCY = 0          # 编码 0 永远是空位；1..S 是 species 列表里的第 1..S 个

# 打包头里用 1 字节记「这是什么晶格」，解包时靠它把模板重建出来。
# 0 留给 custom（只由形状推不回来，必须自带模板）。
KIND_ID = {"custom": 0, "sc": 1, "fcc": 2, "bcc": 3, "hcp": 4}
ID_KIND = {v: k for k, v in KIND_ID.items()}


def _as3(n):
    """允许 n 写成一个整数（立方）或三个整数（各向异性超胞）。"""
    a = np.atleast_1d(np.asarray(n, dtype=int)).ravel()
    if a.size == 1:
        a = np.repeat(a, 3)
    if a.size != 3:
        raise ValueError("n 要么是一个整数，要么是三个整数，收到 {}".format(n))
    if np.any(a < 1):
        raise ValueError("每边的惯用胞数必须 ≥ 1，收到 {}".format(n))
    return tuple(int(x) for x in a)


# ================================================================ 异常
class NotOnLattice(ValueError):
    """结构里有原子对不上格点。带诊断量，方便统计「弛豫到什么程度就拍不进去了」。"""

    def __init__(self, msg, dev=None, n_bad=0, bad=None):
        super().__init__(msg)
        self.dev = dev          # (N,) 每个原子到最近格点的距离，单位=格子
        self.n_bad = n_bad
        self.bad = bad          # 超差原子的序号（前 20 个）


# ================================================================ 模板
class LatticeTemplate:
    """组合信息：网格多大、哪些网格点是真正的格点。所有结构共用一份。

    属性
    ----
    mask         (nx,ny,nz) bool，True = 这个网格点是格点
    grid         (nx,ny,nz)
    site_idx     (M,3) 规范顺序的格点网格坐标（按 i,j,k 字典序，即 np.argwhere 的顺序）
    site_index   (nx,ny,nz) 整型，网格点 -> 它是第几个格点，非格点为 -1
    """

    def __init__(self, mask, kind="custom"):
        m = np.ascontiguousarray(np.asarray(mask, dtype=bool))
        if m.ndim != 3:
            raise ValueError("mask 必须是三维")
        if m.sum() == 0:
            raise ValueError("模板里一个格点都没有")
        self.mask = m
        self.kind = kind
        self.grid = tuple(int(x) for x in m.shape)
        self.site_idx = np.argwhere(m)                     # (M,3)
        sidx = np.full(self.grid, -1, dtype=np.int32)
        sidx[tuple(self.site_idx.T)] = np.arange(len(self.site_idx), dtype=np.int32)
        self.site_index = sidx

    # ------------------------------------------------------------ 构造
    @classmethod
    def sc(cls, n):
        """简单立方：n 个惯用胞/边 -> n³ 网格，全满。

        n 可以是一个整数（立方超胞），也可以是三个整数，例如 `sc((1, 4, 2))`。
        """
        g = _as3(n)
        return cls(np.ones(g, bool), "sc")

    @classmethod
    def bcc(cls, n):
        """体心立方：n 个惯用胞/边 -> (2n)³ 网格，只填 i≡j≡k (mod 2)，占 1/4。

        `bcc(2)` 是 2×2×2 个惯用胞 = 4×4×4 网格 = 16 个格点；
        `bcc((1, 4, 2))` 是同样 16 个格点的 1×4×2 超胞，网格 2×8×4。
        两者格点数相同，靠**网格形状**区分。
        """
        g = tuple(2 * x for x in _as3(n))
        i, j, k = np.indices(g)
        return cls((i % 2 == j % 2) & (j % 2 == k % 2), "bcc")

    @classmethod
    def fcc(cls, n):
        """面心立方：n 个惯用胞/边 -> (2n)³ 网格，只填 (i+j+k) 偶，占 1/2。"""
        g = tuple(2 * x for x in _as3(n))
        i, j, k = np.indices(g)
        return cls((i + j + k) % 2 == 0, "fcc")

    @classmethod
    def of(cls, kind, grid):
        """按 (kind, grid) 把模板重建出来 —— 这正是 pack 头里存的两样东西。"""
        grid = tuple(int(x) for x in grid)
        if len(grid) != 3 or min(grid) < 1:
            raise ValueError("grid 要三个正整数，收到 {}".format(grid))
        if kind == "sc":
            return cls(np.ones(grid, bool), "sc")
        if kind not in ("fcc", "bcc"):
            raise ValueError(
                "模板 {} 只靠形状推不回来（需要自带 mask）".format(kind))
        if any(x % 2 for x in grid):
            raise ValueError("{} 的网格每边必须是偶数，收到 {}".format(kind, grid))
        i, j, k = np.indices(grid)
        if kind == "fcc":
            m = (i + j + k) % 2 == 0
        else:
            m = (i % 2 == j % 2) & (j % 2 == k % 2)
        return cls(m, kind)

    @classmethod
    def hcp(cls):
        """六方最密堆积：正交六方胞 (a, √3a, c)，网格 3×3×2，占 2/6。"""
        i, j, k = np.indices((3, 3, 2))
        m = np.zeros((3, 3, 2), bool)
        m[0, 0, 0] = True
        m[1, 2, 1] = True
        return cls(m, "hcp")

    # ------------------------------------------------------------ 查询
    @property
    def n_sites(self):
        return len(self.site_idx)

    @property
    def n_grid(self):
        return int(self.mask.size)

    @property
    def fill_fraction(self):
        return self.n_sites / self.n_grid

    def key(self):
        """给「同一个模板」一个稳定的短标识，便于库里去重。"""
        return "{}:{}x{}x{}".format(self.kind, *self.grid)

    def describe(self):
        return ("{:<4s} 网格 {}×{}×{} = {:4d} 点   格点 {:4d}   占 {:5.1%}".format(
            self.kind, self.grid[0], self.grid[1], self.grid[2],
            self.n_grid, self.n_sites, self.fill_fraction))

    def __repr__(self):
        return "<LatticeTemplate {}>".format(self.key())

    # ------------------------------------------------------------ MC 用的小工具
    def fractional(self):
        """(M,3) 每个格点在**单位立方**里的分数坐标。"""
        return self.site_idx / np.asarray(self.grid, dtype=float)

    def site_spacing(self):
        """最小的位点间距，单位 = 格子。`from_ase` 的 tol 必须小于它的一半。"""
        g = np.asarray(self.grid, dtype=float)
        f = self.site_idx / g
        d = f[:, None, :] - f[None, :, :]
        d -= np.round(d)                                   # 最小镜像
        r = np.linalg.norm(d * g, axis=2)                  # 换回格子单位
        np.fill_diagonal(r, np.inf)
        return float(r.min())

    def shell(self, n_shell=1, with_multiplicity=False):
        """第 n 近邻壳的格点对，返回 (M, Z) 的格点序号数组。

        距离在最小镜像下算。n_shell=1 就是最近邻 —— bcc 是 8，fcc 是 12，sc 是 6。

        ★ 超胞只有 1 个惯用胞时，最小镜像会把两个方向塌成同一个位点：例如
        bcc 2×2×2 超胞（16 个格点）里，第 2 近邻（立方棱，6 个）只剩 3 个**不同**
        位点，每个的**多重度是 2**。算能量时不能只数位点，要把多重度乘上，
        所以 with_multiplicity=True 会额外返回 (M, Z) 的多重度。

        MC 里算交换能差：`E += sum(mult[i,z] * J[species[i], species[idx[i,z]]])`。
        """
        if n_shell < 1:
            raise ValueError("n_shell 从 1 起数")
        f = self.fractional()
        g = np.asarray(self.grid, dtype=float)
        d = f[:, None, :] - f[None, :, :]
        d -= np.round(d)                                   # 最小镜像
        # ★ 必须乘回网格尺寸：分数坐标下的距离只在各边等高时才和真实距离成比例，
        #   各向异性的超胞（bcc((10,10,5))）不乘就会把近邻壳分错。
        r = np.linalg.norm(d * g, axis=2)
        np.fill_diagonal(r, np.inf)
        vals = np.unique(np.round(r[np.isfinite(r)], 6))
        if len(vals) < n_shell:
            raise ValueError("只剩 {} 个近邻壳".format(len(vals)))
        target = vals[n_shell - 1]

        m = np.abs(r - target) < 1e-6
        out = [np.where(m[i])[0] for i in range(len(f))]
        z = len(out[0])
        if any(len(o) != z for o in out):
            raise ValueError("近邻数不齐，这个模板可能不是好晶格")
        idx = np.array(out, dtype=np.int32)
        if not with_multiplicity:
            return idx
        mult = np.zeros(r.shape, dtype=np.int32)
        for t in itertools.product((-1, 0, 1), repeat=3):
            dd = f[:, None, :] - f[None, :, :] + np.array(t, dtype=float)
            mult += (np.abs(np.linalg.norm(dd * g, axis=2) - target) < 1e-6)
        return idx, mult[np.arange(len(f))[:, None], idx]


# ================================================================ 照片
class Picture:
    """一张照片：模板 + 每个格点上的编码。**不含晶胞**。"""

    def __init__(self, template, code, species=(),
                 _check=True):
        self.template = template
        self.species = list(species)
        self.code = np.ascontiguousarray(np.asarray(code, dtype=np.uint8))
        if self.code.shape != template.grid:
            raise ValueError("照片形状 {} 与模板 {} 不符".format(
                self.code.shape, template.grid))
        if _check:
            bad = self.code[~template.mask]
            if bad.any():
                raise ValueError("非格点上写了东西（模板决定它们必须为空）")
            mx = int(self.code.max()) if self.code.size else 0
            # 符号表可以为空（只当占位矩阵用），这时不校验编码上界。
            if self.species and mx > len(self.species):
                raise ValueError("编码 {} 超出符号表长度 {}".format(
                    mx, len(self.species)))

    # ------------------------------------------------------------ 属性
    def __repr__(self):
        return "<Picture {} {} atoms>".format(self.template.key(), self.n_atoms)

    @property
    def n_atoms(self):
        return int((self.code > VACANCY).sum())

    def values(self):
        """(M,) 只取格点上的编码，规范顺序。"""
        return self.code[tuple(self.template.site_idx.T)]

    def species_at(self):
        """(M,) 每个格点的元素符号；空位是 None。"""
        v = self.values()
        return [self.species[c - 1] if c > 0 else None for c in v]

    def composition(self):
        """{元素: 原子数}，按元素名排序。"""
        out = {}
        for s in self.species_at():
            if s is not None:
                out[s] = out.get(s, 0) + 1
        return dict(sorted(out.items()))

    def formula(self):
        return "".join("{}{}".format(k, v) for k, v in self.composition().items())

    # ------------------------------------------------------------ 转换
    @classmethod
    def from_ase(cls, atoms, template, species=None, tol=0.50):
        """把 ASE 结构拍成照片。

        tol 的单位是**格子**（网格步长），默认 0.50 ≈ 半个最近邻。对 bcc n=2
        一个格子是 a/2 = 1.6425 Å，0.50 格子 = 0.82 Å。

        上限由几何定死：只要 tol < 半个**位点间距**，最近的格点就是唯一的。
        bcc 的位点间距是 a√3/2 = 2.845 Å = 1.732 个格子，所以 tol 最大可以给
        到 0.866；再大就会出现两个格点等距、归属有歧义。fcc 同理（间距 a/√2
        对一个格子是 √2/2... 按模板算即可）。

        判定方式是「找最近的**格点**」而不是「四舍五入到最近网格点」：先按整格
        取整，再在 3×3×3 邻域里筛出真格点取最近的一个。这样即使原子朝某个空位
        偏过去，也不会被错判成空位。
        """
        if species is None:
            species = sorted(set(atoms.get_chemical_symbols()))
        species = list(species)
        n_at = len(atoms)

        grid = np.asarray(template.grid, dtype=float)
        f = atoms.get_scaled_positions(wrap=True)              # (N,3) 分数坐标
        g = f * grid                                           # 网格坐标（浮点）

        r = np.rint(g).astype(np.int64)
        best_d = np.full(n_at, np.inf)
        best = np.zeros((n_at, 3), dtype=np.int64)
        gi = np.asarray(template.grid, dtype=np.int64)
        for off in itertools.product((-1, 0, 1), repeat=3):
            cand = (r + np.array(off)) % gi
            ok = template.mask[cand[:, 0], cand[:, 1], cand[:, 2]]
            if not ok.any():
                continue
            d = np.linalg.norm(g - (r + np.array(off)), axis=1)
            better = ok & (d < best_d)
            best_d[better] = d[better]
            best[better] = cand[better]

        dev = np.where(np.isfinite(best_d), best_d, np.inf)
        bad = np.where(~(dev <= tol))[0]
        if len(bad):
            raise NotOnLattice(
                "{} 个原子对不上格点（最近 {} 个，最差 {:.3f} 格子）。"
                "先是第 {} 号：{:+.3f} 格子".format(
                    len(bad), n_at, float(np.max(dev[bad])), int(bad[0]),
                    float(dev[bad[0]])),
                dev=dev, n_bad=len(bad), bad=bad[:20].tolist())

        # 两个原子不能抢同一个格点
        flat = best[:, 0] * template.grid[1] * template.grid[2] \
            + best[:, 1] * template.grid[2] + best[:, 2]
        if len(np.unique(flat)) != n_at:
            raise NotOnLattice("有原子落在同一个格点上", dev=dev, n_bad=n_at)

        index = {s: i + 1 for i, s in enumerate(species)}
        code = np.zeros(template.grid, dtype=np.uint8)
        for n, s in enumerate(atoms.get_chemical_symbols()):
            if s not in index:
                raise KeyError("符号 {} 不在 species 表里".format(s))
            code[best[n, 0], best[n, 1], best[n, 2]] = index[s]
        return cls(template, code, species)

    def to_ase(self, cell, pbc=True):
        """照片 + 晶胞 -> ASE 结构（**理想几何**）。按格点规范顺序排原子。"""
        cell = np.asarray(cell, dtype=float)
        if cell.shape != (3, 3):
            raise ValueError("cell 要给 3×3 矩阵")
        if Atoms is None:
            raise ImportError("to_ase 需要 ase")
        sites = self.template.site_idx
        vals = self.values()
        keep = vals > 0
        frac = sites[keep] / np.asarray(self.template.grid, dtype=float)
        pos = frac @ cell
        syms = [self.species[v - 1] for v in vals[keep]]
        return Atoms(symbols=syms, positions=pos, cell=cell, pbc=pbc)

    def with_cell(self, a):
        """立方晶胞的便捷写法。"""
        return self.to_ase(np.eye(3) * a)

    # ------------------------------------------------------------ MC 基本动作
    def swap(self, i, j, inplace=False):
        """交换第 i、j 个**格点**上的内容。两个都空或同一个就返回自己。

        MC 的一步就是这个：`new = pic.swap(i, j)`。
        """
        p = self if inplace else self.copy()
        si = tuple(self.template.site_idx[i])
        sj = tuple(self.template.site_idx[j])
        p.code[si], p.code[sj] = self.code[sj], self.code[si]
        return p

    def occupied_sites(self):
        return np.where(self.values() > 0)[0]

    def copy(self):
        return Picture(self.template, self.code.copy(), self.species, _check=False)

    def repeat(self, rep):
        """沿三个方向复制（MC 用大超胞）。返回新照片 + 需要同步的模板。"""
        rep = np.atleast_1d(np.asarray(rep, dtype=int))
        if rep.size == 1:
            rep = np.repeat(rep, 3)
        if np.any(rep < 1):
            raise ValueError("rep 必须是正整数")
        new = LatticeTemplate(np.tile(self.template.mask, rep), self.template.kind)
        code = np.tile(self.code, rep)
        return Picture(new, code, self.species)

    # ------------------------------------------------------------ 打包
    def pack(self, scheme="bits", base=None, with_shape=True, dense=False):
        """把照片压成字节。

        头（`with_shape=True` 时 5 字节，否则 1 字节）
            字节 0    进制 base；最高位 = 1 表示「后面跟着形状头」
            字节 1    晶格种类 | 0x80 表示 dense
            字节 2-4  网格 nx, ny, nz
        载荷
            格点上的编码，按规范顺序排成一个大整数。

        为什么必须带形状：同样 16 个原子的 2×2×2 和 1×4×2 超胞，**格点编码
        完全一样**，只有网格形状（4×4×4 vs 2×8×4）能分开。不带形状的 7 字节
        裸块只能和保管好的模板配对使用。

        ★ 非格点上的 0 不写进载荷：它们由 (种类, 形状) 唯一决定，写下来纯属冗余。
        bcc 有 3/4、fcc 有 1/2 的格点恒为空 —— 这部分在内存里是照片的一部分
        （`pic.code` 就是那张 (4,4,4) 的整照片），在磁盘上则由模板推导。
        `dense=True` 可以把它们也写出来（16 原子五元：11 字节 -> 26 字节）。

        进制默认取 `最大编码 + 1`，即这张照片实际用到的符号数 + 1：二元 2 bit，
        五元 3 bit。进制写在头里，解包不需要外部告诉它符号表有多长。
        上界由元素周期表定死：编码最多 119 种状态，每格点最多 7 bit。
        """
        if with_shape:
            k = KIND_ID.get(self.template.kind, 0)
            if k == 0:
                raise ValueError(
                    "模板种类 {} 只靠形状推不回来：要么用 sc/fcc/bcc，"
                    "要么 pack(with_shape=False) 并自己保管模板".format(
                        self.template.kind))
            if max(self.template.grid) > 255:
                raise ValueError("网格边长 {} 超过 255，头里放不下".format(
                    max(self.template.grid)))
            hdr = bytes([k | (0x80 if dense else 0)] + list(self.template.grid))
        else:
            hdr = b""

        vals = (self.code.ravel() if dense else self.values()).astype(np.int64)
        if scheme == "raw":
            return b"\x00" + hdr + vals.astype(np.uint8).tobytes()

        if base is None:
            base = int(vals.max()) + 1 if vals.size else 2
        base = int(base)
        if not 2 <= base <= 127:
            # 最高位留给「有没有形状头」的标志，所以基数码只能到 127。
            # 元素周期表封死在 119 种状态，够用。
            raise ValueError("进制 {} 不在 2..127 内".format(base))
        if vals.size and int(vals.max()) >= base:
            raise ValueError("编码 {} 超出进制 {}".format(int(vals.max()), base))
        acc = 0
        for v in vals:
            acc = acc * base + int(v)
        nbytes = ((base ** len(vals) - 1).bit_length() + 7) // 8
        flag = 0x80 if with_shape else 0x00
        return bytes([base | flag]) + hdr + acc.to_bytes(nbytes, "big")

    @classmethod
    def unpack(cls, blob, template=None, species=()):
        """解包。带形状头的块可以单独解；不带的自备模板。"""
        b0 = blob[0]
        base = b0 & 0x7F
        if b0 & 0x80:
            if len(blob) < 5:
                raise ValueError("形状头不完整")
            kind = ID_KIND.get(blob[1] & 0x7F)
            dense = bool(blob[1] & 0x80)
            grid = tuple(int(x) for x in blob[2:5])
            if template is None:
                if kind is None:
                    raise ValueError("头里的晶格种类 {} 不认识".format(blob[1] & 0x7F))
                template = LatticeTemplate.of(kind, grid)
            elif tuple(template.grid) != grid:
                raise ValueError("头里的形状 {} 与给定模板 {} 不符".format(
                    grid, template.grid))
            payload = blob[5:]
            n = template.n_grid if dense else template.n_sites
        else:
            if template is None:
                raise ValueError("这块没有形状头，必须给模板")
            dense = False
            payload = blob[1:]
            n = template.n_sites

        if base == 0:                                  # raw：1 字节/格点
            vals = np.frombuffer(payload, dtype=np.uint8).astype(np.int64)
            if len(vals) != n:
                raise ValueError("raw 长度 {} 与预期 {} 个不符".format(len(vals), n))
        else:
            acc = int.from_bytes(payload, "big")
            vals = np.zeros(n, dtype=np.int64)
            for i in range(n - 1, -1, -1):
                acc, vals[i] = divmod(acc, base)
            if acc:
                raise ValueError("打包数据与模板规模不匹配")

        if dense:
            code = vals.reshape(template.grid).astype(np.uint8)
            if code[~template.mask].any():
                raise ValueError("非格点上写了非零编码，这张照片不合法")
            return cls(template, code, species)
        code = np.zeros(template.grid, dtype=np.uint8)
        code[tuple(template.site_idx.T)] = vals
        return cls(template, code, species)

    # ------------------------------------------------------------ 原子序数
    def zcodes(self):
        """(M,) 原子序数 Z（空位是 0）。给 ASE / 别的软件用的自描述写法。"""
        from ase.data import atomic_numbers as _ZN
        lut = np.zeros(len(self.species) + 1, dtype=np.uint8)
        for i, s in enumerate(self.species):
            lut[i + 1] = _ZN[s]
        return lut[self.values()]

    @classmethod
    def from_zcodes(cls, template, z, species):
        """原子序数数组 -> 照片（`zcodes()` 的逆）。"""
        from ase.data import chemical_symbols
        lut = {s: i + 1 for i, s in enumerate(species)}
        z = np.asarray(z, dtype=np.int64)
        vals = np.zeros(len(z), dtype=np.int64)
        for i, zi in enumerate(z):
            if zi:
                s = chemical_symbols[int(zi)]
                if s not in lut:
                    raise KeyError("符号 {} 不在 species 表里".format(s))
                vals[i] = lut[s]
        code = np.zeros(template.grid, dtype=np.uint8)
        code[tuple(template.site_idx.T)] = vals
        return cls(template, code, species)


# ================================================================ 库
class PictureLibrary:
    """一堆照片 + 每条的**初始**晶格参数 + 每原子能量 + 元数据。存成压缩的 .npz。

    每条记录就是用户说的三样东西：**理想结构（照片）、能量、初始晶格参数**。

    ★ 只存初始晶格参数（建模时定的那个 a），不存弛豫后的晶胞，两个理由：

      1. **正确性**。晶格能量模型（CE / LEP）的输入是「理想点阵 + 占位」，预测时
         根本没有弛豫胞可用。把它存进库里，等于把答案那一侧的信息混进了输入——
         CE 的 12.20 meV/atom 就是这么被撤回的：整个增益来自 6 个形变分量，
         而那 6 个数在任何 POSCAR 里都不存在。
      2. **体积**。一个 3×3 晶胞是 9 个 float64 = 72 字节，比照片本身（16 字节）
         大 4 倍多，而它通常只是一个标量 a。775 条实测：存弛豫胞 90.2 字节/条，
         存标量 a 约 14 字节/条。

    ★ **a 只是记录，不参与模型。** 弛豫本来就会把这个参数放开自由调，所以它不决定
    最终结果；存它只是为了知道「这条是在多大的理想点阵上建模的」。因此 `a0` 是可选的
    —— 不给 a 也不给 a0 时，照片、能量、成分、打包全照常工作，只有真的要坐标
    （`atoms()`）时才需要它。非立方胞可以给 3×3 矩阵，那是例外路径。
    """

    def __init__(self, template=None, species=(), a0=None):
        self.templates = [] if template is None else [template]
        self.species = list(species)
        self.a0 = None if a0 is None else float(a0)
        self.tpl_id = []         # 每条属于第几个模板
        self.codes = []          # 每条 (M_k,) uint8，只存格点，长度随尺寸变
        self.offsets = [0]       # 首尾相接存放的位置
        self.a = []              # 每条的初始 a；nan = 用 a0
        self.cells = []          # 非立方胞时才用，None 表示没有
        self.energy = []         # 每原子能量 eV/atom；nan = 还没算
        self.meta = []           # 每个 dict

    def __len__(self):
        return len(self.codes)

    @property
    def template(self):
        """单模板库的快捷方式；混装库请用 `.templates`。"""
        if len(self.templates) != 1:
            raise ValueError("这个库混装了 {} 个模板，请用 .templates".format(
                len(self.templates)))
        return self.templates[0]

    def _tpl_index(self, template):
        for i, t in enumerate(self.templates):
            if t.grid == template.grid and np.array_equal(t.mask, template.mask):
                return i
        self.templates.append(template)
        return len(self.templates) - 1

    def add(self, pic, energy=None, a=None, **meta):
        """加一条。尺寸/晶格可以与前一条不同；新符号会自动并进库的符号表。"""
        a_val, cell = self._norm_lattice(a)
        # 把照片自己的局部编码重映射到库的并集符号表（追加式，老编码不会失效）
        remap = np.zeros(len(pic.species) + 1, dtype=np.uint8)
        for i, s in enumerate(pic.species):
            if s not in self.species:
                self.species.append(s)
            remap[i + 1] = self.species.index(s) + 1
        vals = remap[pic.values()]
        self.codes.append(np.ascontiguousarray(vals, dtype=np.uint8))
        self.tpl_id.append(self._tpl_index(pic.template))
        self.offsets.append(self.offsets[-1] + int(vals.size))
        self.a.append(a_val)
        self.cells.append(cell)
        self.energy.append(np.nan if energy is None else float(energy))
        self.meta.append(meta)
        return len(self.codes) - 1

    @staticmethod
    def _norm_lattice(a):
        if a is None:
            return np.nan, None
        arr = np.asarray(a, dtype=float)
        if arr.shape == ():
            return float(arr), None
        if arr.shape == (3, 3):
            return np.nan, arr
        raise ValueError("a 要么是标量（立方胞），要么是 3×3 矩阵，收到形状 {}".format(
            arr.shape))

    def lattice(self, k):
        """第 k 条的晶格：标量 a（立方胞）或 3×3 矩阵。"""
        if self.cells[k] is not None:
            return self.cells[k]
        if np.isfinite(self.a[k]):
            return float(self.a[k])
        if self.a0 is None:
            raise ValueError("第 {} 条没给 a，库也没有 a0".format(k))
        return self.a0

    def cell_matrix(self, k):
        lat = np.asarray(self.lattice(k), dtype=float)
        return np.eye(3) * lat if lat.ndim == 0 else lat

    def template_of(self, k):
        return self.templates[self.tpl_id[k]]

    def n_sites(self, k):
        return int(self.codes[k].size)

    def grouped(self):
        """{模板序号: [条目序号, ...]}。混尺寸库里按模板分组算近邻/能量。"""
        out = {}
        for k, t in enumerate(self.tpl_id):
            out.setdefault(int(t), []).append(k)
        return out

    def picture(self, k):
        tpl = self.templates[self.tpl_id[k]]
        code = np.zeros(tpl.grid, dtype=np.uint8)
        code[tuple(tpl.site_idx.T)] = self.codes[k]
        return Picture(tpl, code, self.species)

    def atoms(self, k, a=None):
        """照片 + 晶格参数 -> **理想**结构（不是 DFT 弛豫后的几何）。

        这是库里唯一需要 `a` 的地方：不给 a 也没有 a0 时会报错，而照片/能量/成分
        的读取完全不受影响。
        """
        cell = self.cell_matrix(k) if a is None else np.eye(3) * float(a)
        return self.picture(k).to_ase(cell)

    def energies(self):
        """(N,) 每原子能量；还没算的是 nan。"""
        return np.array(self.energy, dtype=float)

    def compositions(self):
        return [self.picture(k).composition() for k in range(len(self))]

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        flat = (np.concatenate(self.codes).astype(np.uint8) if self.codes
                else np.zeros(0, dtype=np.uint8))
        d = dict(
            ntpl=np.int32(len(self.templates)),
            species=np.array(self.species, dtype=str),
            codes=flat,
            offset=np.array(self.offsets, dtype=np.int64),
            tpl_id=np.array(self.tpl_id, dtype=np.int32),
            a=np.array(self.a, dtype=np.float32),
            energy=np.array(self.energy, dtype=float),
            meta=np.array([json.dumps(m, ensure_ascii=False) for m in self.meta]),
            a0=np.float64(np.nan if self.a0 is None else self.a0),
        )
        for i, t in enumerate(self.templates):
            d["mask_{}".format(i)] = t.mask
            d["kind_{}".format(i)] = np.array(t.kind)
        if any(c is not None for c in self.cells):
            d["cells"] = np.array([np.zeros((3, 3)) if c is None else c
                                   for c in self.cells], dtype=float)
        np.savez_compressed(path, **d)
        return path

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        a0 = float(z["a0"]) if "a0" in z.files else np.nan
        lib = cls(None, [str(s) for s in z["species"]],
                  a0=None if not np.isfinite(a0) else a0)
        for i in range(int(z["ntpl"])):
            lib.templates.append(
                LatticeTemplate(z["mask_{}".format(i)], str(z["kind_{}".format(i)])))
        off = np.asarray(z["offset"], dtype=np.int64)
        codes = z["codes"]
        has_cells = "cells" in z.files
        for k in range(len(z["tpl_id"])):
            lib.codes.append(np.ascontiguousarray(codes[off[k]:off[k + 1]]))
            lib.tpl_id.append(int(z["tpl_id"][k]))
            lib.a.append(float(z["a"][k]))
            lib.energy.append(float(z["energy"][k]))
            lib.meta.append(json.loads(str(z["meta"][k])))
            c = z["cells"][k] if has_cells else None
            lib.cells.append(None if (c is None or not np.any(c)) else c)
        lib.offsets = off.tolist()
        return lib


# ================================================================ 自查
def _selftest():
    print("[自查] 三种晶格的模板")
    for tpl in (LatticeTemplate.sc(2), LatticeTemplate.fcc(1),
                LatticeTemplate.bcc(2)):
        print("   " + tpl.describe())
        note = ("   ← 盒子每边只有 2 个格子宽，最小镜像把 +a 和 −a 塌成同一"
                "个位点，所以只数出 3 个" if tpl.grid[0] == 2 else "")
        print("     第1近邻 {:d} 个{}".format(tpl.shell(1).shape[1], note))
    assert LatticeTemplate.sc(2).n_sites == 8
    assert LatticeTemplate.fcc(1).n_sites == 4
    assert LatticeTemplate.bcc(2).n_sites == 16
    # ★ 近邻数只有在超胞够大时才是 8/12/6：1 个惯用胞用最小镜像会退化成 3 个。
    assert LatticeTemplate.bcc(2).shell(1).shape[1] == 8
    assert LatticeTemplate.fcc(2).shell(1).shape[1] == 12
    assert LatticeTemplate.sc(4).shell(1).shape[1] == 6

    # ★ 各向异性的超胞：距离必须在**格子**单位下算，否则壳会分错。
    #   分数坐标下的距离只在各边等高时与真实距离成比例（这是修过的一个真 bug）。
    ta = LatticeTemplate.bcc((8, 8, 4))
    assert ta.shell(1).shape[1] == 8, ta.shell(1).shape
    assert ta.shell(2).shape[1] == 6, ta.shell(2).shape
    print("   各向异性 bcc 8×8×4（网格 16×16×8）：第1壳 8、第2壳 6  ✓")

    print("[自查] 用 ase 建的 bcc 超胞能否原样拍回去")
    a = 3.285
    tpl = LatticeTemplate.bcc(2)
    if bulk is None:
        code = np.zeros(tpl.grid, np.uint8)
        code[tuple(tpl.site_idx.T)] = [1, 2, 3, 4, 5] * 3 + [1]
        pic = Picture(tpl, code, ["Al", "Nb", "Ti", "V", "Zr"])
        print("   （没装 ase，跳过坐标往返，用构造的照片继续）")
    else:
        at = bulk("Ti", "bcc", a=a, cubic=True).repeat((2, 2, 2))
        at.set_chemical_symbols(
            ["Al", "Nb", "Ti", "V", "Zr", "Al", "Nb", "Ti",
             "V", "Zr", "Al", "Nb", "Ti", "V", "Zr", "Al"])
        pic = Picture.from_ase(at, tpl, ["Al", "Nb", "Ti", "V", "Zr"], tol=0.20)
        back = pic.with_cell(2 * a)
        assert back.get_chemical_formula() == at.get_chemical_formula()
        # 逐原子比：照片按格点规范顺序，原结构按 ase 的 repeat 顺序，所以比集合
        ref = sorted(zip(at.get_chemical_symbols(),
                         np.round(at.get_scaled_positions(wrap=True), 8).tolist()))
        got = sorted(zip(back.get_chemical_symbols(),
                         np.round(back.get_scaled_positions(wrap=True), 8).tolist()))
        assert ref == got, "往返后位置或元素不一致"
        print("   16 原子 bcc 往返：元素与分数坐标逐位一致  ✓")
    print("   pack() = {} 字节（含 5 字节头）".format(len(pic.pack())))
    assert Picture.unpack(pic.pack(), tpl, pic.species).values().tolist() \
        == pic.values().tolist()
    # 不带形状的裸块：7 字节，必须配模板
    assert len(pic.pack(with_shape=False)) == 7
    assert Picture.unpack(pic.pack(with_shape=False), tpl,
                          pic.species).values().tolist() == pic.values().tolist()
    print("   pack/unpack 往返（含形状头 / 裸块）  ✓")

    print("[自查] ★ 同样 16 个原子，2×2×2 与 1×4×2 必须分得开")
    shapes = ((2, 2, 2), (1, 4, 2), (4, 1, 2))
    blobs = []
    for cells in shapes:
        t = LatticeTemplate.bcc(cells)
        code = np.zeros(t.grid, np.uint8)
        code[tuple(t.site_idx.T)] = [1, 2, 3, 4, 5] * 3 + [1]
        p = Picture(t, code, ["Al", "Nb", "Ti", "V", "Zr"])
        b = p.pack()
        blobs.append(b)
        # 只给字节、不给模板，也要能把形状解回来
        q = Picture.unpack(b)
        assert tuple(q.template.grid) == t.grid, (q.template.grid, t.grid)
        assert q.values().tolist() == p.values().tolist()
        assert q.code[~q.template.mask].sum() == 0
    assert len(set(blobs)) == len(shapes), "不同形状的照片仍然撞车了"
    print("   {} 个形状 -> {} 种不同的字节，且每个都能单独解回形状  ✓".format(
        len(shapes), len(set(blobs))))
    # dense：把非格点的 0 也写出去，字节变多但内容等价
    t = LatticeTemplate.bcc(2)
    p = Picture.unpack(blobs[0])
    d = p.pack(dense=True)
    assert len(d) > len(blobs[0])
    assert Picture.unpack(d).code.tolist() == p.code.tolist()
    print("   稀疏 {} 字节 / dense {} 字节，解出来是同一张 (4,4,4) 照片  ✓".format(
        len(blobs[0]), len(d)))

    print("[自查] 库：a 只是记录，缺席也能存/读/取能量")
    sp = ["Al", "Nb", "Ti", "V", "Zr"]
    lib = PictureLibrary(tpl, sp)                 # 故意不给 a0
    lib.add(pic, energy=-7.1234, index=1)
    lib.add(pic, energy=-7.4567, index=2, a=3.30)
    assert lib.picture(0).formula() == pic.formula()
    assert np.allclose(lib.energies(), [-7.1234, -7.4567])
    assert abs(lib.lattice(1) - 3.30) < 1e-12
    try:
        lib.lattice(0)
    except ValueError:
        pass                                      # 没 a 也没 a0：只有要坐标时才算错
    else:
        raise AssertionError("缺 a 时 lattice() 应该报错")
    if bulk is not None:                          # 出坐标要 ase
        assert len(lib.atoms(1)) == 16            # 单独给过 a 的那条能出坐标
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.npz")
        lib.save(p)
        lib2 = PictureLibrary.load(p)
        assert len(lib2) == 2
        assert np.allclose(lib2.energies(), [-7.1234, -7.4567]), "能量没读回"
        assert abs(lib2.lattice(1) - 3.30) < 1e-6, "a 没读回"   # a 存 float32
        assert lib2.picture(0).composition() == pic.composition()
        if bulk is not None:
            assert len(lib2.atoms(1)) == 16
    print("   无 a0 的库：成分/能量/单条 a 存取正常，只有要坐标时才要求 a  ✓")

    print("[自查] 编码上界与打包进制")
    assert pic.pack()[0] & 0x7F == len(sp) + 1, "五元照片的进制应是 6"
    p1 = Picture(tpl, pic.code.clip(max=1), ["Al"])    # 只有一种符号 -> 2 种状态
    assert p1.pack()[0] & 0x7F == 2, "一元照片应是 2 进制"
    assert len(p1.pack(with_shape=False)) == 1 + 2, "一元裸块应是 2 字节载荷"
    print("   一元 2 进制（裸块 3 字节）、五元 6 进制（裸块 7 字节）  ✓")
    if bulk is not None:                          # zcodes 查元素表要 ase
        z = pic.zcodes()
        assert z.max() <= 118 and z[z > 0].min() >= 1
        assert Picture.from_zcodes(tpl, z, pic.species).values().tolist() \
            == pic.values().tolist()
        print("   zcodes 往返（1..118）  ✓")

    print("[自查] 混尺寸库：2 / 16 / 54 原子装在一个库里")
    if bulk is None:
        print("   （没装 ase，跳过）")
        print("[自查] 全部通过")
        return
    libm = PictureLibrary(species=["Ti"])       # 空库，符号表随后自动扩充
    a0 = 3.285
    orig = []
    for n, sps in ((1, ["Ti"]), (2, ["Al", "Nb", "Ti", "V", "Zr"]), (3, ["Al", "Zr"])):
        t = LatticeTemplate.bcc(n)
        at = bulk("Ti", "bcc", a=a0, cubic=True).repeat((n, n, n))
        at.set_chemical_symbols([sps[i % len(sps)] for i in range(len(at))])
        origin_comp = at.get_chemical_formula()
        orig.append((at, origin_comp))
        libm.add(Picture.from_ase(at, t, sps, tol=0.20),
                 energy=-7.0 + t.n_sites, a=a0, n=n)
    assert len(libm.templates) == 3, "三个尺寸应该登记成三个模板"
    assert [libm.n_sites(k) for k in range(3)] == [2, 16, 54]
    assert libm.species == ["Ti", "Al", "Nb", "V", "Zr"], libm.species
    for k, (at, comp) in enumerate(orig):
        assert libm.atoms(k).get_chemical_formula() == comp, k
        assert sorted(libm.atoms(k).get_chemical_symbols()) \
            == sorted(at.get_chemical_symbols()), k
    assert np.allclose(libm.energies(), [-7.0 + 2, -7.0 + 16, -7.0 + 54])
    assert libm.grouped() == {0: [0], 1: [1], 2: [2]}
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "m.npz")
        libm.save(p)
        m2 = PictureLibrary.load(p)
        assert len(m2) == 3 and len(m2.templates) == 3
        assert [m2.n_sites(k) for k in range(3)] == [2, 16, 54]
        assert np.allclose(m2.energies(), libm.energies())
        for k, (at, comp) in enumerate(orig):
            assert m2.atoms(k).get_chemical_formula() == comp, k
            assert m2.picture(k).values().tolist() == libm.picture(k).values().tolist()
    print("   2/16/54 原子共库：成分、能量、模板、存取全一致  ✓")

    print("[自查] 全部通过")


if __name__ == "__main__":
    _selftest()

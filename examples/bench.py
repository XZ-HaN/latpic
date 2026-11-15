# -*- coding: utf-8 -*-
"""latpic 的三项代价：字节数、CSRO、MC 每步。

只依赖 numpy，不需要 ase / torch，所以在任何机器上都能复现 README 里的数字。

    python examples/bench.py

五段：
  1  单张照片的 pack() 字节数（2 / 16 / 54 / 128 / 1000 原子）
  2  PictureLibrary 每条的字节数（含能量、含初始 a）
  3  CSRO：Xᵀ S_L X 每个结构的耗时
  4  MC：PairModel 每步耗时与"每步只重算几十行"
  5  断点状态：一张照片 vs 特征缓存
"""
from __future__ import annotations

import os
import pickle
import sys
import tempfile
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from latpic import LatticeTemplate, Picture, PictureLibrary      # noqa: E402
from latpic_csro import CSROEngine                               # noqa: E402
from latpic_mc import FeatureEngine, MonteCarlo, PairModel       # noqa: E402

FIVE = ["Al", "Nb", "Ti", "V", "Zr"]
SEED = 20261026


def head(t):
    print()
    print("=" * 72)
    print(t)
    print("=" * 72)


def rand_codes(tpl, species, rng):
    """等原子分数的随机占位；编码 0 = 空位，1..S = species 下标 + 1。"""
    M = tpl.n_sites
    q, r = divmod(M, len(species))
    base = np.concatenate([np.repeat(np.arange(1, len(species) + 1), q),
                           np.arange(1, r + 1)])
    return base[rng.permutation(M)].astype(np.uint8)


def pic_of(tpl, codes, species):
    code = np.zeros(tpl.grid, np.uint8)
    code[tuple(tpl.site_idx.T)] = codes
    return Picture(tpl, code, species)


def sec1():
    head("1  单张照片 pack() 的字节数")
    print("  {:<10s}{:>8s}{:>10s}{:>12s}{:>12s}".format(
        "原子数", "格点数", "裸块 B", "自描述 B", "dense B"))
    rng = np.random.default_rng(SEED)
    for cells in ((1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4), (10, 10, 5)):
        tpl = LatticeTemplate.bcc(cells)
        codes = rand_codes(tpl, FIVE, rng)
        pic = pic_of(tpl, codes, FIVE)
        print("  {:<10d}{:>8d}{:>10d}{:>12d}{:>12d}".format(
            tpl.n_sites, int(np.prod(tpl.grid)),
            len(pic.pack(with_shape=False)), len(pic.pack()),
            len(pic.pack(dense=True))))
    print("  裸块 = 载荷 + 1 字节进制；自描述 = 再加 4 字节（晶格种类 + 网格 nx,ny,nz）")


def sec2():
    head("2  PictureLibrary 每条多少字节")
    tpl = LatticeTemplate.bcc(2)                     # 16 原子
    rng = np.random.default_rng(SEED)
    n = 1000
    lib = PictureLibrary(tpl, FIVE, a0=3.285)
    for _ in range(n):
        lib.add(pic_of(tpl, rand_codes(tpl, FIVE, rng), FIVE),
                energy=float(-7.0 - rng.random()), a=3.285 + 0.01 * rng.random())
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "lib.npz")
        lib.save(p)
        size = os.path.getsize(p)
    print("  {} 条 16 原子五元结构（含每原子能量 + 初始 a + 模板）：".format(n))
    print("    .npz 共 {:,.0f} B  =  {:.1f} B/条".format(size, size / n))
    print("  对照：同样 {} 个 POSCAR 约 {:,.0f} B（每个 ~1 200 B）".format(
        n, n * 1200))
    print("  为什么这么小：只存 M 个格点编码（16 B 未压缩），不存稠密网格；"
          "a 是标量，不存 3×3 胞。")


def sec3():
    head("3  CSRO：Xᵀ S_L X")
    tpl = LatticeTemplate.bcc((10, 10, 5))           # 1000 原子
    n_shells = 2
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    eng = CSROEngine(tpl, n_shells=n_shells)
    t_eng = time.perf_counter() - t0
    print("  引擎（近邻关系只算一次）：{:.3f} s   每壳 Z = {}".format(
        t_eng, [round(s[2], 1) for s in eng.shells]))
    n = 300
    t = []
    for _ in range(n):
        pic = pic_of(tpl, rand_codes(tpl, FIVE, rng), FIVE)
        t0 = time.perf_counter()
        eng.alpha(pic)
        t.append(time.perf_counter() - t0)
    t = np.asarray(t)
    print("  {} 个 1000 原子结构（到第 {} 壳）：均值 {:.2f} ms  中位 {:.2f} ms".format(
        n, n_shells, 1e3 * t.mean(), 1e3 * np.median(t)))
    print("  成本与结构无关，只与格点数 M 和壳数有关：一次 (M,S)ᵀ·(M,S) 矩阵乘。")


def sec4():
    head("4  MC：一个建议的代价")
    tpl = LatticeTemplate.bcc((10, 10, 5))
    n_shells = 5
    rng = np.random.default_rng(SEED)
    codes = rand_codes(tpl, FIVE, rng)
    t0 = time.perf_counter()
    eng = FeatureEngine(tpl, FIVE, n_shells=n_shells)
    t_eng = time.perf_counter() - t0
    model = PairModel(0.02 * rng.standard_normal((n_shells, 5, 5)))
    print("  {}（引擎 {:.3f} s）".format(eng.describe(), t_eng))

    t0 = time.perf_counter()
    for _ in range(200):
        eng.features_from_codes(codes)
    t_full = (time.perf_counter() - t0) / 200

    mc = MonteCarlo(eng, model, temperature=1000.0, seed=SEED)
    n = 20000
    t0 = time.perf_counter()
    run = mc.run(codes, n, verbose=False)
    t_step = (time.perf_counter() - t0) / n
    print("  一次全量特征重算            {:8.2f} ms".format(1e3 * t_full))
    print("  一个 MC 步（特征 + 能量）    {:8.1f} μs   -> {:.0f} 步/秒".format(
        1e6 * t_step, 1.0 / t_step))
    print("  一步只重算 {:,.0f} / {} 行；等于一次全量重算的 1/{:.0f}".format(
        run.n_touched / n, tpl.n_sites, t_full / t_step))
    print("  接受率 {:.1%}".format(run.accept_rate))


def sec5():
    head("5  断点状态要存多少")
    tpl = LatticeTemplate.bcc((10, 10, 5))
    rng = np.random.default_rng(SEED)
    codes = rand_codes(tpl, FIVE, rng)
    pic = pic_of(tpl, codes, FIVE)
    eng = FeatureEngine(tpl, FIVE, n_shells=5)
    F = eng.features(pic).astype(np.float32)
    E = np.zeros(tpl.n_sites, np.float32)
    blob_pic = pic.pack()
    blob_codes = codes.tobytes()
    blob_feat = zlib.compress(pickle.dumps({"features": F, "energies": E}, 4), 9)
    print("  照片 pack()（自描述，能直接还原）  {:9,d} B".format(len(blob_pic)))
    print("  裸 codes 数组                      {:9,d} B".format(len(blob_codes)))
    print("  特征 + 逐原子能量（zlib+pickle）   {:9,d} B".format(len(blob_feat)))
    print("  -> 断点只存 codes 就够了，特征在恢复时按近邻表重算（{:.0f} ms 量级）。"
          .format(1e3 * _t(eng, codes)))
    p2 = Picture.unpack(blob_pic, species=FIVE)
    assert p2.values().tolist() == codes.tolist()
    print("  ✓ pack/unpack 往返无损")


def _t(eng, codes):
    t0 = time.perf_counter()
    eng.features_from_codes(codes)
    return time.perf_counter() - t0


def main():
    sec1()
    sec2()
    sec3()
    sec4()
    sec5()
    print("\n完成。")


if __name__ == "__main__":
    main()

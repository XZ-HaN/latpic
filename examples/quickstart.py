# -*- coding: utf-8 -*-
"""latpic 的最小可用示例。只依赖 numpy。

    python examples/quickstart.py

五段，对应 README 的「快速上手」：
  1  建模板、造一张照片、打包/解包
  2  存库：理想结构 + 能量 + 初始 a
  3  CSRO：一次矩阵乘
  4  MC：状态就是一个 uint8 数组
  5  退火走到穷举已知的基态
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from latpic import LatticeTemplate, Picture, PictureLibrary          # noqa: E402
from latpic_csro import CSROEngine, csro_report                      # noqa: E402
from latpic_mc import FeatureEngine, MonteCarlo, PairModel, anneal   # noqa: E402

FIVE = ["Al", "Nb", "Ti", "V", "Zr"]


def pic_of(tpl, codes, species):
    code = np.zeros(tpl.grid, np.uint8)
    code[tuple(tpl.site_idx.T)] = codes
    return Picture(tpl, code, species)


def main():
    rng = np.random.default_rng(0)

    # ---------------------------------------------------------- 1
    print("[1] 模板 + 照片 + 打包")
    tpl = LatticeTemplate.bcc(2)                       # 2×2×2 惯用胞
    print("   " + tpl.describe())
    codes = np.array([1, 2, 3, 4, 5] * 3 + [1], np.uint8)
    pic = pic_of(tpl, codes, FIVE)
    blob = pic.pack()
    print("   照片 {}   公式 {}   pack() = {} 字节".format(
        pic, pic.formula(), len(blob)))
    back = Picture.unpack(blob, species=FIVE)          # 形状头自描述，不用给模板
    assert back.template.grid == tpl.grid
    assert back.values().tolist() == pic.values().tolist()
    print("   解包回来：{}   ✓ 无损".format(back.template.describe()))
    print("   裸块（不带形状头）= {} 字节，必须配同一个模板".format(
        len(pic.pack(with_shape=False))))

    # ---------------------------------------------------------- 2
    print("\n[2] 库：理想结构 + 能量 + 初始 a")
    lib = PictureLibrary(tpl, FIVE, a0=3.285)
    for k in range(200):
        c = np.array(rng.permutation(codes), np.uint8)
        lib.add(pic_of(tpl, c, FIVE), energy=-7.0 - 0.01 * k, a=3.285, idx=k)
    print("   {} 条，能量 {} … {}".format(
        len(lib), np.round(lib.energies()[0], 4), np.round(lib.energies()[-1], 4)))
    with tempfile.TemporaryDirectory() as td:
        p = lib.save(os.path.join(td, "lib.npz"))
        size = os.path.getsize(p)
        lib2 = PictureLibrary.load(p)
    assert np.allclose(lib2.energies(), lib.energies())
    print("   存成 .npz 共 {} B = {:.1f} B/条，读回后能量逐位一致  ✓".format(
        size, size / len(lib)))
    print("   注意库里只存**理想结构 + 能量 + 初始 a**，不存弛豫后的晶胞。")

    # ---------------------------------------------------------- 3
    print("\n[3] CSRO：α = 1 − n_ij /(M·Z·c_i·c_j)，n_ij = Xᵀ S_L X")
    tpl10 = LatticeTemplate.bcc(4)                     # 128 原子
    eng_c = CSROEngine(tpl10, n_shells=2)
    c = np.random.default_rng(1).integers(1, 6, tpl10.n_sites).astype(np.uint8)
    print(csro_report(eng_c.alpha(pic_of(tpl10, c, FIVE))))
    print("   规则：非对角 α<0 = 异类结伴（有序），α>0 = 异类回避（偏聚）；"
          "对角线符号相反。")

    # ---------------------------------------------------------- 4
    print("\n[4] MC：状态就是一个 (M,) 的 uint8 数组")
    eng = FeatureEngine(tpl, FIVE, n_shells=4)
    model = PairModel(0.02 * np.random.default_rng(2).standard_normal((4, 5, 5)))
    mc = MonteCarlo(eng, model, temperature=1000.0, swap_ratio=1.0, seed=0)
    run = mc.run(codes, 2000, verbose=False)
    print("   " + run.summary().replace("\n", "\n   "))

    # ---------------------------------------------------------- 5
    print("\n[5] 退火走到穷举已知的基态（16 原子，Al–Ti，只有第一壳有对势）")
    tpl2 = LatticeTemplate.bcc(2)
    sp = ["Al", "Ti"]
    eng2 = FeatureEngine(tpl2, sp, n_shells=4)
    J = np.zeros((4, 2, 2))
    J[0] = [[0.0, -0.20], [-0.20, 0.0]]                # 第一壳异类成键有利
    model2 = PairModel(J)
    c0 = np.array([1, 2] * 8, np.uint8)
    run2, _ = anneal(eng2, model2, c0,
                     [3000, 2000, 1500, 1000, 700, 500, 350, 250, 200, 150],
                     3000, verbose=False)
    e_b2 = model2(pic_of(tpl2, np.where(tpl2.site_idx[:, 0] % 2 == 0, 1, 2),
                         sp), eng2)
    print("   退火终点 E = {:.6f} eV   B2 构型 E = {:.6f} eV   差 {:.2e} eV".format(
        run2.total, e_b2, abs(run2.total - e_b2)))
    print("   ✓ 走到 B2 基态")

    print("\n完成。")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""latpic_mc —— 在 latpic 照片格式上做蒙特卡洛。

跟 LEP 的 `lep/mc.py` 的区别只有一条，但它是根本性的：

    MC 的**全部状态**就是一个 (M,) 的 uint8 数组。

没有 ase.Atoms、没有 DataFrame、没有 base64/pickle 的特征缓存。一次交换只动两个
字节；一次建议只需要重算「被改动格点及其邻居」那几十行的特征。

模块
----
features : FeatureEngine —— 近邻表 + 特征矩阵，以及**逐行**重算
models   : LinearModel / PairModel / TorchModel
mc       : MonteCarlo、anneal

典型用法
--------
    from latpic import LatticeTemplate, Picture
    from latpic_mc import FeatureEngine, PairModel, MonteCarlo

    tpl = LatticeTemplate.bcc(10)
    eng = FeatureEngine(tpl, ["Al", "Nb", "Ti", "V", "Zr"], n_shells=5)
    mc  = MonteCarlo(eng, PairModel(J, n_shells=5, n_species=5), temperature=800)
    run = mc.run(pic, n_steps=200_000)
    run.picture          # 结果还是一张照片
"""

from .features import FeatureEngine, picture_from_codes          # noqa: F401
from .models import LinearModel, PairModel, TorchModel           # noqa: F401
from .mc import MonteCarlo, MCRun, anneal                        # noqa: F401

__all__ = [
    "FeatureEngine", "picture_from_codes",
    "LinearModel", "PairModel", "TorchModel",
    "MonteCarlo", "MCRun", "anneal",
]

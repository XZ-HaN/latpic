# -*- coding: utf-8 -*-
"""能量模型 —— 只要能「吃特征、吐每个原子的能量」，就能进 MC。

统一接口（所有模型都是这一个）：

    model.energies(F) -> (n,)      每个原子的能量，F 是 (n, n_feat) 特征
    model(pic, engine) -> float    整张照片的总能量

跟 LEP 的耦合点在 `TorchModel`：它直接调用 `lep.train.NNmodel` 的
`forward(x, lengths)`（x 形状 (1, n, F)，返回 (1, n)），所以用户已经训好的
模型权重不用改一行就能拿来跑 MC。
"""

import numpy as np


class LinearModel:
    """`E_atom = W[s_i] · F_i + b`。

    `w` 有两种形状：

    * `(n_feat,)` —— 所有原子共用一套权重（最常见的线性回归）；
    * `(K, n_feat)` —— **按原子自己的元素取一行**（`s_i` 从 F 的第 0 组 one-hot 取）。

    第二种是必须的：对势 `J[s_i, s_j]` 依赖原子 i 自己的元素，共用一套权重
    表达不了。因为 F 的第 0 组就是自己元素的 one-hot，所以「按行取权重」的
    查找是 O(n·K)，不引入任何额外状态。

    注：空位（自己那一组全 0）取第 0 行的壳权重，但在位能项恒为 0，因为
    F 的第 0 组全 0。
    """

    def __init__(self, w, b=0.0):
        self.w = np.asarray(w, np.float64)
        if self.w.ndim not in (1, 2):
            raise ValueError("w 应当是 (n_feat,) 或 (K, n_feat)")
        self.b = float(b)

    @property
    def n_feat(self):
        return self.w.shape[-1]

    @property
    def n_species(self):
        return self.w.shape[0] if self.w.ndim == 2 else 1

    def energies(self, F):
        F = np.asarray(F, np.float64)
        if self.w.ndim == 1:
            return F @ self.w + self.b
        K = self.w.shape[0]
        sid = np.argmax(F[:, :K], axis=1)          # 自己元素的列号
        return np.einsum("nk,nk->n", F, self.w[sid]) + self.b

    def __call__(self, pic, engine):
        return float(self.energies(engine.features(pic)).sum())

    def __repr__(self):
        return "LinearModel({}×{} 参数{})".format(
            self.n_species, self.n_feat, "，按元素分行" if self.w.ndim == 2 else "")


class PairModel(LinearModel):
    """纯对势：`E = ½ Σ_L Σ_{i≠j} J_L[s_i, s_j] · 多重度`。

    Parameters
    ----------
    J : array (n_shells, K, K)
        `J[L-1, a, b]` 是第 L 壳上 a–b 对的能量。**只需要填上三角，会自动对称化**；
        对角元（同类对）不会被转成两倍，写成 `J[a,a]` 就是「a–a 一个键的能量」。
    on_site : array (K,) 或 None
        「每个原子在自己的位置上」的那部分能量（在线性模型里就是第 0 组的权重）。
    """

    def __init__(self, J, on_site=None, n_species=None):
        J = np.asarray(J, np.float64)
        if J.ndim != 3:
            raise ValueError("J 的形状应当是 (壳数, K, K)")
        n_shells, K, K2 = J.shape
        if K != K2:
            raise ValueError("J 的后两维必须相等")
        if n_species is not None and n_species != K:
            raise ValueError("n_species 与 J 的维度不一致")
        # 对称化：只填了一半的调用方式也应当给出对称的物理
        J = 0.5 * (J + J.transpose(0, 2, 1))
        self.J = J
        self.n_shells = n_shells            # n_species 由 LinearModel 的属性给出

        n_feat = (n_shells + 1) * K
        W = np.zeros((K, n_feat), np.float64)          # W[自己的元素, 特征列]
        if on_site is not None:
            # 第 0 组是自己元素的 one-hot，所以只有对角元起作用
            W[np.arange(K), np.arange(K)] = np.asarray(on_site, np.float64).ravel()
        for L in range(n_shells):
            # E_atom_i = Σ_s ½ J[s_i, s] n_{L,s}
            W[:, (L + 1) * K:(L + 2) * K] = 0.5 * J[L]
        super().__init__(W, 0.0)

    def __repr__(self):
        return "PairModel({} 壳 × {}×{} 对势)".format(
            self.n_shells, self.n_species, self.n_species)


class TorchModel:
    """包一层 LEP 的 `NNmodel`（或任何 `forward(x, lengths)` 同签名的模块）。

    LEP 的约定：`x` 形状 `(n_struct, n_max_atom, n_feat)`，`lengths` 是每个结构
    的原子数，返回 `(n_struct, n_max_atom)` 的逐原子能量（padding 被 mask 成 0）。
    这里永远只放**一个**结构进去，所以 lengths = [n]。

    Parameters
    ----------
    module : torch.nn.Module
    device : str 或 torch.device 或 None（None = 跟随 module 的参数位置）
        显式给 device 时会把 module 的参数**搬过去**（否则前向会报
        "found at least two devices, cpu and cuda:0"）。
    autocast : bool
        CUDA + bf16 支持时开 autocast（与 LEP 的 `_update_energy` 一致）。
        注意：开 autocast 会让增量重算的能量与全量重算差一点（bf16 精度），
        验证「增量 == 全量」时请关掉。
    n_threads : int 或 None
        **MC 里几乎一定要设成 1。** 每步只有几十行的前向，torch 默认按核数开
        OpenMP 线程，线程同步的开销会反过来占掉全部时间 —— 在 104 核的节点上
        实测默认线程数 41 ms/步、单线程 0.1 ms/步，差 400 倍。注意这是
        **进程级全局设置**，会同时影响进程里其它 torch 代码。
    """

    def __init__(self, module, device=None, autocast=False, n_threads=None):
        import torch                                        # 延迟：ce 环境没有 torch
        self._torch = torch
        self.module = module
        if device is None:
            try:
                self.device = next(module.parameters()).device
            except (StopIteration, AttributeError):
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            module.to(self.device)
        if n_threads is not None:
            torch.set_num_threads(int(n_threads))
        self.autocast = bool(autocast)
        self.module.eval()

    def energies(self, F):
        torch = self._torch
        x = torch.as_tensor(np.asarray(F, np.float32), device=self.device)[None]
        lengths = torch.tensor([x.shape[1]], device=self.device)
        with torch.no_grad():
            if self.autocast and x.is_cuda and torch.cuda.is_bf16_supported():
                with torch.cuda.amp.autocast():
                    out = self.module(x.half(), lengths)
                out = out.float()
            else:
                out = self.module(x, lengths)
        return out[0].detach().cpu().numpy()

    def __call__(self, pic, engine):
        return float(self.energies(engine.features(pic)).sum())

    def __repr__(self):
        return "TorchModel({})".format(type(self.module).__name__)

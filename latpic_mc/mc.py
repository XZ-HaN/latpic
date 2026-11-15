# -*- coding: utf-8 -*-
"""MonteCarlo —— 照片上的 Metropolis 采样。

跟 LEP 的 `MonteCarloSampler` 的差别（每一条都是为了去掉一个耦合点）：

| LEP                                    | 这里                                  |
|----------------------------------------|---------------------------------------|
| 状态 = `ase.Atoms` + DataFrame + 特征缓存 | 状态 = `(M,)` 的 uint8 `codes`         |
| 每步 `cached_features.clone()` × 2      | 不复制任何 (M,F) 数组                  |
| 特征按「改动的原子 + 邻居」重算          | 同左，但用预存的 `touch` 表，O(|rows|·Z) |
| 断点续算靠 `atoms.info` 里的 pickle     | `MCRun.codes` 就是全部状态，存 11 字节  |

能量账本只有三样东西：`F`（每行特征）、`E_atom`（每行能量）、`total`。
每步只重算被改动格点 **及其邻居** 的那几十行。
"""

import time

import numpy as np

from .features import picture_from_codes

KB = 8.617333262e-5          # eV/K，与 LEP 的 mc.py 一致


class MCRun:
    """一次 MC 的结果。**全部状态就是 `codes`** —— 拿它就能重建照片。"""

    def __init__(self, codes, template, species, total, e_atom,
                 trace=None, n_steps=0, n_accepted=0, n_swaps=0, n_touched=0,
                 wall=None, temperatures=None):
        self.codes = codes
        self.template = template
        self.species = list(species)
        self.total = float(total)
        self.e_atom = e_atom
        self.trace = trace if trace is not None else []
        self.n_steps = n_steps
        self.n_accepted = n_accepted
        self.n_swaps = n_swaps
        self.n_touched = n_touched
        self.wall = wall
        self.temperatures = temperatures or []

    @property
    def picture(self):
        return picture_from_codes(self.template, self.codes, self.species)

    @property
    def accept_rate(self):
        return self.n_accepted / self.n_steps if self.n_steps else 0.0

    def composition(self):
        """{元素: 个数}"""
        return {s: int((self.codes == i + 1).sum())
                for i, s in enumerate(self.species)}

    def summary(self):
        out = ["  终点总能量 {:12.4f} eV   （{:.4f} eV/atom）".format(
            self.total, self.total / max(len(self.codes), 1)),
            "  步数 {}，接受率 {:.1%}".format(self.n_steps, self.accept_rate)]
        if self.wall:
            out.append("  墙钟 {:.3f} s，即 {:,.0f} 步/秒，{:.1f} μs/步".format(
                self.wall, self.n_steps / self.wall, 1e6 * self.wall / max(self.n_steps, 1)))
            out.append("  平均每步重算 {:.0f} 行特征（共 {} 行）".format(
                self.n_touched / max(self.n_steps, 1), len(self.codes)))
        out.append("  成分 " + "  ".join("{}={}".format(k, v)
                                        for k, v in self.composition().items()))
        return "\n".join(out)


class MonteCarlo:
    """Metropolis 采样器。

    Parameters
    ----------
    engine : FeatureEngine
    model  : 任何有 `.energies(F)` 的对象（`LinearModel` / `PairModel` / `TorchModel`）
    temperature : float
        K。`0` 或 `None` = 只走下降（贪心淬火）。
    swap_ratio : float
        建议是「两个格点交换」的概率；其余是「把一个格点换成随机元素」
        （后者**改变成分**，与 LEP 的 `random_replace` 同义）。
    """

    def __init__(self, engine, model, temperature=300.0, swap_ratio=0.5, seed=0):
        self.engine = engine
        self.model = model
        self.temperature = temperature
        self.swap_ratio = float(swap_ratio)
        self.rng = np.random.default_rng(seed)
        self._M = engine.n_sites
        self._K = engine.K
        self._n_species = len(engine.species)

    # ------------------------------------------------------------------ 内部
    def _state(self, pic_or_codes):
        eng = self.engine
        if isinstance(pic_or_codes, np.ndarray):
            codes = np.array(pic_or_codes, np.uint8)
            if codes.shape != (self._M,):
                raise ValueError("codes 应当是 ({},) 的数组".format(self._M))
        else:
            if pic_or_codes.template.grid != eng.template.grid:
                raise ValueError("照片与引擎的模板不是同一个")
            codes = np.array(pic_or_codes.values(), np.uint8)
        OH = eng.onehot_from_codes(codes)
        F = eng.features_from_oh(OH)
        return codes, OH, F

    def _propose(self):
        """返回 (i, j, is_swap)。j = -1 表示「把 i 换成另一个元素」。"""
        M = self._M
        if self.rng.random() < self.swap_ratio:
            i = int(self.rng.integers(M))
            j = int(self.rng.integers(M - 1))
            if j >= i:
                j += 1
            return i, j, True
        return int(self.rng.integers(M)), -1, False

    def _new_row(self, i):
        """给 i 抽一个新元素（编码 1..S）。replace 用。"""
        return int(self.rng.integers(self._n_species)) + 1

    def _delta_features(self, rows, i, j, D):
        """只改动 i（旧→新）与 j（新→旧）时，rows 这几行的特征增量。

        rows 第 L 组第 c 列 = Σ_z mult·OH[idx[L][r,z], c]。i 与 j 的 OH 行互为交换，
        所以两行都等于 +D 或 −D，按「r 在 i 上的邻居多重度和」减去「在 j 上的」
        再乘 D 就是增量。O(|rows|·Z)，不做任何 (M,F) 复制。
        """
        eng = self.engine
        n = len(rows)
        delta = np.zeros((n, eng.n_feat), np.float64)
        gi = eng.nb_idx[rows]                             # (n, Z总)
        gm = eng.nb_mult[rows]
        w = ((gi == i) * gm)                              # (n, Z总)，按壳分块
        if j >= 0:
            w -= (gi == j) * gm
        w = np.add.reduceat(w, eng.shell_starts, axis=1)  # 每块求和 -> (n, 壳)
        # 各壳的特征组是连续存放的（组 L = 列 [L·K, (L+1)·K)），一次 reshape 就够
        delta[:, eng.K:] = (w[:, :, None] * D).reshape(n, -1)
        return delta

    # ------------------------------------------------------------------ 主循环
    def run(self, pic_or_codes, n_steps, temperature=None, record_every=0,
            progress=0, verbose=True):
        """跑 `n_steps` 次建议。返回 `MCRun`。

        record_every : int
            >0 时每这么多步记一次总能量（默认只在接受时记，避免列表爆炸）。
        progress : int
            >0 时每这么多步打一行进度。
        """
        T = self.temperature if temperature is None else temperature
        beta = 0.0 if not T else 1.0 / (KB * T)

        codes, OH, F = self._state(pic_or_codes)
        E_atom = np.asarray(self.model.energies(F), np.float64)
        total = float(E_atom.sum())

        eng = self.engine
        K, nf = eng.K, eng.n_feat
        inv_touch = eng.touch

        n_acc = n_swaps = n_touch = 0
        trace = []
        t0 = time.perf_counter()
        rnd = self.rng.random

        for step in range(n_steps):
            # 抽一个「真的会改变东西」的建议：同种元素互换、换成本身都是空操作，
            # 直接计入步数会把接受率压低（5 元体系里 1/5 的交换建议是空的）。
            for _retry in range(8):
                i, j, is_swap = self._propose()
                if is_swap:
                    if codes[i] != codes[j]:
                        break
                else:
                    new_i = self._new_row(i)
                    if new_i != codes[i]:
                        break
            else:
                continue
            if is_swap:
                n_swaps += 1
                new_i, new_j = codes[j], codes[i]
            else:
                new_j = None

            rows = np.unique(np.concatenate(
                (inv_touch[i], inv_touch[j] if j >= 0 else inv_touch[i][:0])))

            D = (eng.onehot_vector(codes[j]) if is_swap else eng.onehot_vector(new_i)) \
                - eng.onehot_vector(codes[i])
            delta = self._delta_features(rows, i, j, D)
            F_new = F[rows] + delta
            F_new[:, :K] = OH[rows]                       # 组 0：自己（下面再改 i/j）
            if is_swap:
                F_new[rows == i, :K] = OH[j]
                F_new[rows == j, :K] = OH[i]
            else:
                F_new[rows == i, :K] = eng.onehot_vector(new_i)

            E_new = np.asarray(self.model.energies(F_new), np.float64)
            old = E_atom[rows]
            dE = float(E_new.sum() - old.sum())

            # ★ beta == 0（T = 0）必须是「只接受下降」，不能靠 exp(0) = 1 —— 那会变成
            #   什么上坡都接受的随机行走。
            ok = dE <= 0.0 or (beta > 0.0 and rnd() < np.exp(-beta * dE))
            if ok:
                F[rows] = F_new
                E_atom[rows] = E_new
                total += dE
                codes[i] = new_i
                if is_swap:
                    codes[j] = new_j
                    OH[i], OH[j] = OH[j].copy(), OH[i].copy()
                else:
                    OH[i] = 0.0
                    OH[i, new_i - 1] = 1.0
                n_acc += 1
                if record_every <= 0:
                    trace.append((step + 1, total))
            n_touch += len(rows)

            if record_every > 0 and (step + 1) % record_every == 0:
                trace.append((step + 1, total))
            if progress > 0 and (step + 1) % progress == 0:
                if verbose:
                    print("  步 {:>9,}   E = {:12.4f}   接受率 {:5.1%}".format(
                        step + 1, total, n_acc / (step + 1)), flush=True)

        wall = time.perf_counter() - t0
        run = MCRun(codes, eng.template, eng.species, total, E_atom,
                    trace=trace, n_steps=n_steps, n_accepted=n_acc,
                    n_swaps=n_swaps, n_touched=n_touch, wall=wall,
                    temperatures=[T] if T else [0.0])
        run.F = F              # 增量维护出来的特征；调试/验证时拿来跟全量重算对账
        if verbose:
            print(run.summary(), flush=True)
        return run


def anneal(engine, model, pic_or_codes, temperatures, steps_per_T,
           swap_ratio=0.5, seed=0, record_every=0, verbose=True):
    """温度阶梯退火。返回 (MCRun, 每次温度后的总能量列表)。

    状态在各级温度之间**连续传递**（不重新初始化），所以每一级都比上一级更省。
    """
    mc = MonteCarlo(engine, model, temperature=temperatures[0],
                    swap_ratio=swap_ratio, seed=seed)
    state = pic_or_codes
    es = []
    run = None
    t_all = 0.0
    for k, T in enumerate(temperatures):
        if verbose:
            print("  [温度 {}/{}] T = {:g} K，{} 步".format(
                k + 1, len(temperatures), T, steps_per_T), flush=True)
        run = mc.run(state, steps_per_T, temperature=T,
                     record_every=record_every, verbose=False)
        es.append(run.total)
        t_all += run.wall
        state = run.codes
        if verbose:
            print("      E = {:12.4f} eV   接受率 {:5.1%}   {:.2f} s".format(
                run.total, run.accept_rate, run.wall), flush=True)
    run.wall = t_all
    run.temperatures = list(temperatures)
    return run, es

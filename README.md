# latpic — lattice pictures

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Dependencies](https://img.shields.io/badge/dependencies-numpy%20only-green.svg)](#installation)
[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

A **lattice picture** is the occupancy of a fixed crystal lattice written as a
small integer grid — one byte per grid point, `0` meaning "no atom". Everything
about the geometry (which grid points are lattice sites, how big the cell is) is
stored once, separately, and shared by every structure.

```
16-atom bcc cell, five elements:   pack() -> 11 bytes      (a POSCAR is ~1.2 kB)
the same thing as a .npz library:  19 bytes per entry      (structure + energy + a)
```

Because the lattice geometry is shared, two structures of the same cell are
compared by comparing a handful of bytes, and Monte Carlo state is literally a
`(M,)` `uint8` array.

---

## Why

Lattice models — cluster expansions, (LE)²P / LEP, Bragg–Williams, lattice MC —
are all defined on the same input:

> **an ideal lattice + an occupancy of its sites → a property (energy)**

Three consequences follow, and latpic is built around them:

1. **The relaxed geometry must not enter.** If you store the DFT-relaxed cell
   alongside the occupancy, you have put information from the *answer* side into
   the *input* side. (This is not hypothetical: a whole round of our own numbers
   had to be retracted because the features were computed on relaxed cells. The
   gain came entirely from six strain components that exist in no POSCAR.)
2. **The geometry is redundant.** A 3×3 conventional cell of a cubic material is
   nine float64s — 72 bytes — to describe what is usually one scalar `a`, and
   the site positions are implied by the lattice type and the cell size.
3. **Size changes, format should not.** A 2-atom cell and a 1000-atom cell should
   live in the same container, because they are the same *kind* of object.

## The idea

A structure is *on the lattice* if

> (fractional coordinate of the atom) × (conventional cell) = (a basis site)

Splitting that conventional cell into a regular grid and keeping the points that
land on basis sites gives the template:

| lattice | grid | mask | fill |
|---|---|---|---|
| simple cubic | `n³` | all points | 1/1 |
| fcc | `(2n)³` | `i+j+k` even | 1/2 |
| bcc | `(2n)³` | `i≡j≡k (mod 2)` | 1/4 |
| hcp | `3×3×2` | basis `(0,0,0)` and `(1/3,2/3,1/2)` | 2/6 |

Three layers, stored separately:

```
LatticeTemplate   which grid points are sites          -> shared by the whole library
Picture           what sits on each site (uint8 grid)  -> one per structure
cell              3x3 matrix or a single cubic edge     -> one per structure
```

Four properties that follow, and that the code asserts in its own self-test:

1. **A picture is always ideal geometry.** Displacements from relaxation are
   discarded on the way in — that is the point, not a limitation.
2. **Non-site grid points are always 0**, decided by the template, so they are
   never written to disk.
3. **Size is a property of a picture, not of the format.** Templates are
   registered on demand, codes are stored back-to-back, so 2-, 16- and 54-atom
   structures coexist in one library.
4. **The encoding is bounded by the periodic table**: `0` = vacancy, `1..S` =
   the library's symbol table, at most 119 states ⇒ at most 7 bits per site,
   forever. `pack()` writes the radix it used into the header, so a binary
   picture is 1 bit/site and a five-element one is 3 bit/site automatically.

## Installation

```bash
pip install numpy          # that is the whole dependency
```

`ase` is optional and only imported for `Picture.from_ase`, `Picture.to_ase` and
`zcodes()`. `torch` is optional and only needed for `latpic_mc.TorchModel`.
The format core runs on numpy alone.

```bash
git clone https://github.com/XZ-HaN/latpic.git
cd latpic
python -X utf8 latpic.py                 # self-test: templates, packing, library
python -X utf8 examples/quickstart.py    # end-to-end demo
```

## Quickstart

```python
from latpic import LatticeTemplate, Picture, PictureLibrary

tpl = LatticeTemplate.bcc(2)                       # 2x2x2 conventional cells = 16 sites
pic = Picture.from_ase(atoms, tpl, species=["Al", "Nb", "Ti", "V", "Zr"], tol=0.20)

pic.formula()                                      # 'Al4Nb3Ti3V3Zr3'
pic.code                                           # (4,4,4) uint8, non-sites are 0

blob = pic.pack()                                  # 11 bytes, self-describing
back = Picture.unpack(blob, species=["Al", "Nb", "Ti", "V", "Zr"])
back.values().tolist() == pic.values().tolist()    # True - lossless

atoms_ideal = pic.to_ase(cell_matrix)              # ideal geometry, canonical order
```

A library — the only thing a lattice-energy model needs to be trained on:

```python
lib = PictureLibrary(tpl, ["Al", "Nb", "Ti", "V", "Zr"], a0=3.285)
lib.add(pic, energy=-7.1234, a=3.285, name="sqs-001")
lib.save("ideal.npz")                              # ~19 bytes per entry
lib2 = PictureLibrary.load("ideal.npz")
lib2.atoms(0)                                      # ideal ASE atoms
```

Short-range order, one matrix multiply per structure:

```python
from latpic_csro import CSROEngine, csro_report
eng = CSROEngine(tpl, n_shells=2)
print(csro_report(eng.alpha(pic)))                 # {(a,b): alpha}
```

Monte Carlo, where the entire state is a `uint8` array:

```python
from latpic_mc import FeatureEngine, PairModel, MonteCarlo, anneal
eng = FeatureEngine(tpl, ["Al", "Nb", "Ti", "V", "Zr"], n_shells=4)
mc = MonteCarlo(eng, PairModel(J), temperature=1000.0, swap_ratio=1.0)
run = mc.run(pic, 200_000)
run.picture                                        # the result is a picture again
```

## API

### `LatticeTemplate`

| | |
|---|---|
| `sc(n)`, `bcc(n)`, `fcc(n)`, `hcp()` | constructors; `n` is an int or `(nx,ny,nz)` |
| `of(kind, grid)` | rebuild from the two things `pack()` stores |
| `.grid`, `.n_grid`, `.n_sites`, `.fill_fraction` | size of the grid / of the lattice |
| `.site_idx`, `.site_index` | `(M,3)` site coordinates / `(nx,ny,nz)` reverse map |
| `.fractional()` | `(M,3)` fractional coordinates in the unit cube |
| `.site_spacing()` | smallest site–site distance, in grid units |
| `.shell(k, with_multiplicity=False)` | `(M,Z)` neighbour indices for shell `k`, plus multiplicities |
| `.key()`, `.describe()` | short id / human-readable line |

### `Picture`

| | |
|---|---|
| `Picture.from_ase(atoms, tpl, species=None, tol=0.50)` | snap a structure onto the lattice |
| `.to_ase(cell)`, `.with_cell(a)` | back to ASE, **ideal** geometry |
| `.values()`, `.code`, `.n_atoms` | `(M,)` site codes / the dense grid |
| `.composition()`, `.formula()` | `{'Al': 4, ...}` / `'Al4Nb3Ti3V3Zr3'` |
| `.swap(i,j)`, `.repeat(rep)`, `.occupied_sites()` | MC-level moves |
| `.pack(...)`, `.unpack(...)` | bytes (see below) |
| `.zcodes()`, `.from_zcodes(...)` | real atomic numbers, for interop |

### `PictureLibrary`

`add(pic, energy=None, a=None, **meta)`, `picture(k)`, `atoms(k)`, `energies()`,
`compositions()`, `grouped()`, `save(path)`, `load(path)`, `templates`.

The library stores exactly three things per entry: **the ideal structure, the
per-atom energy, and the initial lattice parameter**. It deliberately does *not*
store relaxed cells (see *Why*). `a` is a record, not a model input: without it
everything still works except `atoms(k)` (which needs coordinates).

### `latpic_csro`

`CSROEngine(template, n_shells=2, with_vacancy=True)` → `.counts(pic)`,
`.alpha(pic, n_shells=None, cumulative=True)`, and `csro_report(alpha)`.

Warren–Cowley α `= 1 − n_ij / (M·Z·c_i·c_j)`, with `n_ij = Xᵀ S_L X`, `X` the
`(M,S)` one-hot of the occupancy and `S_L` the shell adjacency of the template —
which does not depend on who sits where. Reading rule: for `i ≠ j`, `α < 0`
means the pair occurs *more* often than random (ordering), `α > 0` less often
(clustering). The diagonal has the opposite sign.

### `latpic_mc`

`FeatureEngine(template, species, n_shells=5, with_vacancy=False)` builds the
`(l+1) × n_elements` descriptor of (LE)²P/LEP — column layout identical to
`lep.StructureAnalyzer.create_atomic_environment_df`, so an existing network
drops straight in via `TorchModel`.

`LinearModel(w, b)`, `PairModel(J, on_site=None)`, `TorchModel(module, ...)`,
`MonteCarlo(engine, model, temperature, swap_ratio)`, `anneal(...)`, `MCRun`.

## Byte layout

`Picture.pack(scheme="bits", base=None, with_shape=True, dense=False)`

```
byte 0        radix (bits 0..6); bit 7 = 1  ->  a shape header follows
shape header  byte 1   lattice kind (0 custom, 1 sc, 2 fcc, 3 bcc, 4 hcp)
                        bit 7 = 1 ->  the payload is dense (all grid points)
              bytes 2-4  nx, ny, nz
payload       site codes in canonical order, one big integer in base `radix`
```

- `radix` defaults to `max(code) + 1`, i.e. the number of symbols this picture
  actually uses plus the vacancy. It is capped at 127 because bit 7 of byte 0 is
  the flag — and 119 elements + vacancy fits.
- **The shape header is not optional decoration.** 16 atoms in a `2×2×2` and in
  a `1×4×2` supercell have *identical* site codes; only the grid shape tells them
  apart. Without it, two different structures pack to the same bytes.
- `dense=False` (default) skips the non-site zeros, which the template implies.
  For bcc that is 3/4 of the grid: `11` bytes instead of `26` for 16 atoms.
- The **species table is not in the blob**. `unpack` needs it from you. The blob
  is self-describing about *shape*, not about *chemistry*.

## Measured cost

From `examples/bench.py` on one CPU core (Windows 11, Python 3.12, numpy 2.5).
Absolute times are machine-dependent; the ratios are the point.

| atoms | grid points | bare block | self-describing | dense |
|---:|---:|---:|---:|---:|
| 2 | 8 | 2 B | 6 B | 7 B |
| 16 | 64 | 7 B | **11 B** | 26 B |
| 54 | 216 | 19 B | 23 B | 75 B |
| 128 | 512 | 43 B | 47 B | 171 B |
| 1000 | 4000 | 325 B | 329 B | 1298 B |

| what | measured |
|---|---|
| library (1000 entries, structure + energy + `a`) | **19.2 B/entry** |
| CSRO, 1000 atoms, 2 shells | 0.26 ms/structure |
| MC step, 1000 atoms, 5 shells, `PairModel` | 61 µs → 16 k steps/s |
| … of which feature rows recomputed | 75 of 1000 (1/24 of a full recompute) |
| MC breakpoint state, 1000 atoms | 329 B (vs 15.3 kB for cached features) |

## Caveats

- **Perfect lattices only.** `from_ase` snaps each atom to its nearest *site*
  and throws the displacement away. A relaxed structure still goes in as long as
  every atom is within `tol`; otherwise `NotOnLattice` is raised, carrying the
  per-atom deviation so you can measure "how relaxed is too relaxed".
- **`tol` is in grid units and is bounded by geometry.** It must stay below half
  the site spacing: for bcc the spacing is `a√3/2 = 1.732` grid units, so `tol`
  can go up to 0.866 and no further. Snapping is to the nearest *site*, not the
  nearest *grid point* — otherwise an atom displaced toward a vacancy would be
  assigned to the vacancy.
- **A small cell cannot resolve far shells.** Minimum image folds `+a` and `−a`
  onto the same site: on a two-grid-wide box (`sc(2)`, `fcc(1)`) the first shell
  is 3 distinct sites, not 6 or 12, and the 16-atom `bcc(2)` cell resolves only
  four shells — its second shell is 3 distinct sites of multiplicity 2.
  `shell(k, with_multiplicity=True)` returns those multiplicities: **multiply by
  them**, or bonds are undercounted by a factor of up to 4 (`fcc(1)`).
- **Do not put relaxed cells in a library.** See *Why*. It is not a storage
  trick; it silently leaks the target into the input.
- Limits: grid edge ≤ 255 and radix ≤ 127 for a self-describing blob; 0 is
  always the vacancy and cannot be used as a species.

## Related

The descriptor implemented in `latpic_mc` is the one used by **(LE)²P**
(Layer Environment Lattice Energy Potential): the local chemical environment of
a site is the elemental composition of its successive neighbour shells, an
`(l+1) × n_e` descriptor that grows linearly with the number of elements.
A picture is the natural state representation for annealing on that model: an
exchange touches two bytes, and only the ~75 rows inside the touched shells need
recomputing.

## License

GPL-3.0, see [LICENSE](LICENSE).

---

## 中文简介

**latpic** 把「固定晶格上的占位」存成一张三维整数照片：每个网格点一个字节，
`0` 表示没有原子；哪些网格点是真格点、晶胞多大，由模板与晶胞另存、全库共用。

- 三层：`LatticeTemplate`（模板，共用）/ `Picture`（照片，每结构一张）/ `cell`（晶胞，每结构一份）。
- 16 原子五元 bcc 照片 `pack()` **11 字节**（POSCAR 约 1.2 kB），带能量的库约 **19 字节/条**。
- 照片**永远是理想点阵几何**；弛豫位移在拍进去时丢掉——对晶格能量模型这正是对的输入。
- 尺寸是**照片的属性**而非格式的属性：2 / 16 / 54 / 1000 原子可以同库混装。
- 编码受元素周期表封顶（≤119 态 ⇒ ≤7 bit/格点），进制写在包里，二元自动 1 bit。
- `latpic_csro` 用一次矩阵乘 `Xᵀ S_L X` 算 Warren–Cowley 参数；`latpic_mc` 的 MC
  状态就是一个 `(M,)` 的 `uint8` 数组，每步只重算被改动格点及其邻居的几十行。
- 注意：`tol` 必须小于半个位点间距（bcc 上限 0.866 格）；吸附到最近的**格点**
  而不是最近的**网格点**；一个惯用胞宽的超胞会把 ±a 塌成同一个位点，近邻要乘多重度；
  包里**不含元素表**，`unpack` 时符号表要自己给。

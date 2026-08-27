# ToF-SIMS Hyperspectral Analysis Pipeline

A reproducible Python pipeline for Time-of-Flight Secondary Ion Mass Spectrometry
(ToF-SIMS) hyperspectral imaging. It reads the instrument's non-standard BMP
exports, decomposes the ion-count cube into chemical endmembers, segments the
specimen spatially, and tests statistically whether those segments are real.

Developed on a fossilised fly specimen (*Fossil Fly*), but nothing in the code is
specific to it: the channel count, pixel grid, polarity grouping and number of
endmembers are all derived from the data or set in one config file.

---

## Quick start

```bash
git clone <your-repo-url>
cd tofsims-pipeline

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python scripts/run_pipeline.py
```

That runs all six stages over `data/raw/` and writes every array, table and figure
to `data/processed/`.

## Results at a glance

NMF resolves the specimen into distinct chemical endmembers, which the segmentation
then maps spatially:

![NMF decomposition](docs/figures/05_fig2_nmf_decomposition.png)

Every mass channel separates the segments at large effect size (peak
epsilon-squared 0.73, all p < 0.001):

![Statistical validation](docs/figures/05_fig4_statistical_validation.png)

The remaining report figures are in [`docs/figures/`](docs/figures/). Outputs are not
committed — regenerate them all with one command.

---

## Processing your own data

This is the part most likely to matter to you. **In the common case you do not edit
any code** — you drop files in and run one command.

### 1. Put your `.bmp` exports somewhere

```bash
python scripts/run_pipeline.py --raw-dir /path/to/my_data --output-dir results/my_run
```

Or copy them into `data/raw/` and just run `python scripts/run_pipeline.py`.

### 2. That's usually it

The pipeline works out the rest by itself:

| Decision | How it is made |
|---|---|
| Which images form the analysis cube | The most common pixel grid in the dataset |
| Which polarity is the primary group | Whichever dominates that grid |
| Channel order | Sorted by m/z, so runs are reproducible |
| Number of NMF endmembers | The elbow of the reconstruction-error curve |
| Number of segments | Matches the endmember count |
| HDBSCAN `min_cluster_size` | Scaled to 2% of the foreground pixels |
| Images on a different grid | Treated as a secondary acquisition and resampled for the cross-polarity comparison |

A dataset with three channels at 96×128 and a different naming convention runs
through the same command unchanged — that case is covered by the test suite.

### 3. Adjust the config only if you need to

Everything lives in [`configs/pipeline_config.yaml`](configs/pipeline_config.yaml).
The settings you are most likely to touch:

```yaml
io:
  num_channels: 4        # interleaved planes in the export
  channel_index: 1       # which plane holds the ion counts
  header_bytes: 54

metadata:
  fields:                # regex per field; a field that matches nothing becomes null
    polarity: '(?:^|[\s_\-])(neg(?:ative)?|pos(?:itive)?)(?:[\s_\-]|$)'
    mass: '(\d+\.?\d*\s*±\s*\d+\.?\d*\s*u)'

analysis:
  target_shape: auto     # or [640, 640] to pin it
  polarity: auto         # or "Neg" / "Pos"

decomposition:
  nmf:
    n_components: auto   # or a fixed integer
```

If your filenames follow a different convention, change the `metadata.fields`
regexes. Each field is matched independently, so an unusual filename loses only the
fields it genuinely lacks rather than failing the run.

---

## Stages

Each stage reads its inputs from disk and writes its outputs there, so any stage can
be re-run on its own after a config change.

```bash
python scripts/run_pipeline.py --stage decompose     # just re-run the decomposition
python scripts/run_pipeline.py --stage segment,stats # or several
python scripts/run_pipeline.py --list-stages
```

| Stage | Does | Key outputs |
|---|---|---|
| `load` | Parses the BMP exports, recovers metadata | `fossilfly_clean_stack.npz`, `metadata.json`, `01_*.png` |
| `overview` | Channel composites and correlation structure | `02a_*.png` |
| `decompose` | PCA, NMF, UMAP | `02b_*.npz`, `02b_*.png` |
| `segment` | 4 segmentations + agreement metrics | `fossilfly_segments.npz`, `03_*.png` |
| `stats` | Kruskal-Wallis, Dunn's post-hoc, colocalization | `04_statistical_results.csv`, `04_*.png` |
| `figures` | Publication-style master figures | `05_fig*.png` |

Useful flags:

```bash
--nmf-k 4        # override the endmember count
--no-umap        # skip UMAP and HDBSCAN (much faster)
--config path    # use a different config file
```

---

## Method notes

**Reading the files.** The exporter writes a standard 54-byte BMP header claiming 64
bits per pixel, then dumps interleaved `uint16` planes. PIL and OpenCV both refuse or
misread this, so [`src/preprocessing/load_images.py`](src/preprocessing/load_images.py)
parses the payload directly, taking the dimensions from the header rather than
assuming a fixed grid.

**Why three decompositions.** PCA gives variance structure but its signed components
are hard to read chemically. NMF is additive and non-negative, so each pixel is a sum
of endmember contributions — physically meaningful for ion counts. UMAP captures
non-linear structure the linear methods miss. They are run on a shared preprocessing
so the results are comparable pixel-for-pixel.

**Why four segmentations.** Unsupervised segment boundaries are a hypothesis. The
pipeline reports pairwise ARI/NMI between methods and cluster purity *against a
majority-class baseline*, rather than reporting a single partition as fact. On the
bundled dataset the methods largely disagree (ARI ≈ 0.09 between the dominant-endmember
and K-Means partitions) — that low agreement is itself a finding about how separable
the chemistry is, and the figures state it rather than hide it.

**Why non-parametric statistics.** Ion-count distributions are heavily zero-inflated
and non-normal, so Kruskal-Wallis is used per channel with Dunn's post-hoc and
multiple-comparison correction. With ~48,000 pixels almost any difference reaches
significance, so effect size (epsilon-squared) is reported alongside every p-value and
is what the interpretation rests on.

---

## Repository layout

```
├── configs/pipeline_config.yaml   # every tunable parameter
├── data/
│   ├── raw/                       # instrument .bmp exports
│   └── processed/                 # generated arrays, tables, figures
├── src/
│   ├── config.py                  # config loading and path resolution
│   ├── pipeline.py                # stage orchestration
│   ├── preprocessing/             # BMP parsing, metadata, normalisation, I/O
│   ├── features/                  # channel selection, PCA/NMF/UMAP
│   ├── segmentation/              # clustering and agreement metrics
│   ├── stats/                     # significance testing, colocalization
│   ├── viz/                       # all figure generation
│   └── utils/                     # paths, constants, formatting
├── notebooks/                     # thin wrappers over src/, for exploration
├── scripts/run_pipeline.py        # CLI entry point
├── tests/                         # unit + end-to-end regression tests
└── docs/legacy/                   # superseded exploratory scripts, kept for reference
```

The notebooks call the same functions as the CLI, so they cannot drift from it.

---

## Tests

```bash
pytest                    # everything
pytest tests/test_units.py  # fast unit tests only
```

Two groups matter:

- **`TestBundledDatasetRegression`** pins the published numbers — 48,294 foreground
  pixels, NMF rank 3, reconstruction error 145738.2, peak effect size 0.731 — so an
  accidental change to the analysis is caught rather than silently absorbed.
- **`TestUnseenDataset`** builds a synthetic dataset with a different pixel grid,
  channel count and naming scheme, then runs the whole pipeline over it with the stock
  config. This is the test that backs the claim that someone else's data will work.

---

## Requirements

Python 3.9+ and the packages in [`requirements.txt`](requirements.txt): numpy, pandas,
scipy, matplotlib, scikit-learn (≥1.3, for `sklearn.cluster.HDBSCAN`),
scikit-posthocs, umap-learn, PyYAML, pytest.

UMAP is optional at runtime — if `umap-learn` is missing, the pipeline logs it and
skips the UMAP and HDBSCAN steps rather than failing.

---

## Known limitations

- Endmember ordering is not stable across datasets or NMF re-fits, so segments are
  labelled generically (`Segment 1 (EM1)`). Supply `channel_semantics.segment_labels`
  in the config to give them chemical names for a specific dataset.
- The secondary-polarity comparison uses the first secondary image when several are
  present.
- Chemical identity of a mass channel is not inferred; the optional
  `channel_semantics.masses` map is used for figure labels only.

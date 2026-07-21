# HOPE-PHY: Test-Time Continuum Memory for 6G Channel Prediction

**Target venue:** IEEE Transactions on Wireless Communications (TWC) / IEEE TMLCN.

HOPE-PHY is a self-modifying, continuum-memory neural channel predictor for
non-stationary 6G channels. Parameter groups ("memory blocks") update at
geometrically spaced frequencies matched to the channel's own temporal
structure (per-slot fading / mobility & shadowing / scenario regime), with a
Titans-style surprise-gated test-time memory update and an EMA slow anchor.

## Claims the code must support
1. **Matched-protocol accuracy** on the LLM4CP QuaDRiGa benchmark (TDD + FDD,
   UMa NLOS, P=16 -> L=4, K=48 RBs, 10-100 km/h): beat LLM4CP / Transformer /
   Mamba-class baselines on NMSE (dB) per velocity.
2. **Streaming non-stationarity** (the novelty stage): scenario-transition
   streams (UMa -> UMi -> highway, cross-dataset DeepMIMO/Sionna) where frozen
   models degrade; HOPE-PHY's test-time continuum memory adapts online with
   zero pilot overhead. Report degradation depth + recovery time + steady NMSE.
3. **Causal ablation** (our signature methodology): remove timescale
   separation / surprise gating / EMA anchor one at a time; show each is
   load-bearing.
4. **Theory:** N-timescale SA stability of the memory recursion + dynamic
   regret bound vs. drift budget (paper-side; validated empirically in
   notebooks/).

## Repository map
```
hope-phy/
├── configs/                  # YAML configs (composable: data/model/train/ttt)
│   ├── base.yaml
│   ├── data/     llm4cp.yaml | deepmimo.yaml | sionna_cdl.yaml | synthetic.yaml
│   ├── model/    rnn.yaml | lstm.yaml | gru.yaml | cnn.yaml | transformer.yaml
│   │             | mamba.yaml | llm4cp.yaml | hope_phy.yaml
│   ├── train/    default.yaml | continual.yaml
│   └── ttt/      off.yaml | naive_sgd.yaml | surprise_cms.yaml
├── src/hope_phy/
│   ├── data/                 # datasets, loaders, normalisation, drift streams
│   │   ├── llm4cp_dataset.py     # LLM4CP released .mat dataset (anchor benchmark)
│   │   ├── synthetic.py          # AR(1)+multipath synthetic gen (pipeline dev/CI)
│   │   ├── streams.py            # streaming scenario-transition protocol
│   │   ├── deepmimo_dataset.py   # [Day 6] ray-traced scenarios
│   │   ├── sionna_dataset.py     # [Day 6] Sionna CDL/UMi generation
│   │   └── transforms.py         # complex<->real, per-antenna norm, windowing
│   ├── models/
│   │   ├── baselines/            # rnn.py (RNN/LSTM/GRU), cnn.py, transformer.py,
│   │   │                         # mamba.py, llm4cp_wrapper.py
│   │   └── cms/                  # OUR METHOD
│   │       ├── memory_block.py       # frequency-f_k memory block
│   │       ├── continuum.py          # CMS: K blocks, geometric update schedule
│   │       ├── titans_memory.py      # surprise-gated test-time deep memory
│   │       └── hope_phy.py           # full model: backbone + CMS + TTT hooks
│   ├── train/                # trainer.py, losses.py (NMSE loss), schedulers.py
│   ├── ttt/                  # test-time adaptation engine, surprise gating,
│   │                         # update-frequency clocks, safeguards (reset/trust)
│   ├── continual/            # EWC, replay buffer, naive fine-tune baselines
│   ├── metrics/              # nmse.py (dB), se_ber.py (spectral eff., BER proxy)
│   ├── eval/                 # matched-protocol eval, streaming eval, stats tests
│   └── utils/                # seeding, logging (wandb/tensorboard), registry
├── scripts/                  # entry points: train.py, eval_matched.py,
│                             # eval_streaming.py, download_data.py, ablate.py
├── tests/                    # shape/NMSE/clock unit tests (run in CI)
├── notebooks/                # theory validation, figures
├── docker/                   # reproducible CUDA image
└── results/                  # checkpoints, logs, paper figures (git-ignored)
```

## Environment
```bash
conda env create -f environment.yml   # or: pip install -e ".[dev]"
conda activate hope-phy
pytest tests/                          # smoke test the pipeline
python scripts/train.py model=lstm data=synthetic   # end-to-end dry run (no dataset needed)
```

## 14-day execution plan
| Day | Deliverable |
|-----|-------------|
| 1   | Env + repo + synthetic pipeline + NMSE metrics + unit tests green |
| 2   | LLM4CP dataset downloaded, loader verified, LSTM/GRU/Transformer baselines training |
| 3   | Baselines reproduce sane NMSE-vs-velocity curves (TDD); LLM4CP wrapper w/ released weights |
| 4   | CMS memory blocks + continuum schedule implemented; offline-trained HOPE-PHY v0 |
| 5   | Titans surprise-gated test-time update + safeguards; TTT engine working |
| 6   | DeepMIMO + Sionna generation; streaming scenario-transition protocol |
| 7   | **Gate:** streaming eval — HOPE-PHY vs frozen + naive-FT. Need >=1 dB advantage trend |
| 8   | Continual baselines (EWC/replay); hyperparameter sweep on update frequencies |
| 9   | Full matched-protocol table (TDD+FDD, all velocities, n seeds) |
| 10  | Ablations: no-separation / no-surprise / no-anchor / single-block |
| 11  | Scale study (model size sweep — GPU moat) + noise robustness + few-shot |
| 12  | Statistical tests (paired permutation, Holm), all tables final |
| 13  | Paper figures (matplotlib, IEEE style) + regret-bound empirical validation |
| 14  | Buffer: repo cleanup, seeds re-run, README for open-source release |

## Data
- **LLM4CP QuaDRiGa** (anchor): download links in the LLM4CP repo
  (github.com/liuboxun/LLM4CP). Place under `data/llm4cp/`.
- **DeepMIMO**: `pip install deepmimo`, scenarios downloaded via script.
- **Sionna**: generated locally (GPU), configs under `configs/data/`.

All results directories are git-ignored; every experiment logs its resolved
config + git commit hash for exact reproducibility.

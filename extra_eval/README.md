# Extra benchmarks for Robin3D

Evaluation of the **released Robin3D checkpoint** (`ckpt_for_robin3d_final_performance/ckpt_01.pth`
from the authors' Google Drive) on all seven benchmarks: the four the paper reports
(**ScanRefer, Multi3DRefer, Scan2Cap, SQA3D**, re-run as a check) and the ones it does not
(**ScanQA** with EM, **Real-3DQA**, **Beacon3D**: ScanNet with the released features,
3RScan / MultiScan with features rebuilt the same way). No training; the model, ScanNet
features and prompts are the authors' own.

## Run

```bash
cp env.sh.example env.sh                       # fill in paths (+ OPENAI_API_KEY for Beacon3D QA)
bash extra_eval/build_env_cu128.sh             # venv: py3.10, torch 2.7.1+cu128 (Blackwell OK)
bash extra_eval/download.sh                    # ckpt + ScanNet Mask3D/Uni3D/DINOv2 features + val annos
PREFIX_CACHE=1 bash extra_eval/run_extra_eval.sh   # inference, all visible GPUs (37 min on 2 RTX PRO 6000)
PREDS_DIR=outputs/extra_eval_<ts> bash extra_eval/score.sh   # CPU; Java for pycocoevalcap
bash extra_eval/aligned/run_score_aligned.sh   # CPU; cross-baseline re-scoring, see *Aligned scoring*
```

Beacon3D 3RScan / MultiScan and XR-QA / XR-SceneCaption have their own steps (sections below).

Without `PREFIX_CACHE=1` the run is the upstream code path (2 h 38 min); see *Faster
evaluation*. Google Drive sometimes refuses a file for a while ("too many accesses");
`download.sh` skips finished files, so just run it again later.

Changes to upstream code (all needed only to run evaluation outside the authors' machine):
- `dataset/base_dataset.py` reads the Vicuna tokenizer path from `$LLAMA`.
- `tasks/shared_utils.py` loads the checkpoint with `weights_only=False` (it pickles the
  config; torch >= 2.6).
- `tasks/train.py` skips its online scorer for the new sets and for the `check_*` aliases of
  ScanRefer / Multi3DRefer / Scan2Cap, which `extra_eval/score.py` scores instead. Robin3D's
  scorers in `utils/eval.py` read the ScanRefer attribute file, the Scan2Cap corpus and the
  unique/multiple lookup from the authors' disk; `score.py` points them at local files.
- `utils/eval.py`: the ScanRefer Unique / Multiple lines, commented out upstream, are back on.
- `models/`: the opt-in fast path (*Faster evaluation*), off by default.

## Setup check

Robin3D's `OBJ_lm_head` (the logits of the `<OBJxxx>` tokens) is frozen and is not in the
checkpoint, so it comes from the seeded random init (seed 42, rank 0 broadcast by DDP). To make
sure this run rebuilt the same model, the paper's benchmarks are re-run and compared with the
authors' released predictions (`preds_epoch1_step5146_*.json`, the run that produced
`ckpt_01`). Scoring those released predictions with `score.py` gives exactly the numbers of
the authors' `train.log` (column 2), so the scorer is the same; columns 1 and 3 are this run.

| | ours (upstream path) | authors' log | `PREFIX_CACHE=1` | identical generations (upstream / cache) |
|---|---|---|---|---|
| ScanRefer Acc@0.25 / @0.5 | 60.76 / 55.01 | 60.81 / 55.07 | 60.78 / 55.06 | 99.5 / 99.6 % |
| ScanRefer Unique @0.25 / @0.5 | 88.74 / 82.43 | 88.80 / 82.49 | 88.69 / 82.38 | |
| ScanRefer Multiple @0.25 / @0.5 | 54.41 / 48.78 | 54.46 / 48.85 | 54.45 / 48.86 | |
| Multi3DRefer F1@0.25 / @0.5 | 64.90 / 59.75 | 64.90 / 59.73 | 64.92 / 59.74 | 98.1 / 98.3 % |
| Scan2Cap CIDEr@0.5 / B-4@0.5 | 86.60 / 38.33 | 87.22 / 38.38 | 85.92 / 38.08 | 88.1 / 85.9 % |
| Scan2Cap CIDEr@0.25 / B-4@0.25 | 91.89 / 40.32 | 92.52 / – | 91.15 / 40.10 | |
| SQA3D test EM / EM-R | 56.86 / 59.73 | 56.89 / 59.76 | 56.46 / 59.36 | 98.8 / 98.3 % |
| ScanQA CIDEr / B-4 | 94.76 / 12.92 | 94.80 / 12.67 | 94.51 / 12.75 | 96.4 / 96.0 % |

The remaining differences are bf16 near-ties in the beam search (other GPU, torch and
flash-attn versions than the authors'), which compound over long outputs, hence Scan2Cap.
Scan2Cap is scored like upstream: 2022 val (scene, object) pairs against the 2068-object
val corpus, through Robin3D's own `calc_scan2cap_score` (corpus objects without a val item
score as empty captions); Multi3DRefer over all 11120 val descriptions.

## Results (`results/`; `PREFIX_CACHE=1` run in `results_prefixcache/`)

Paper benchmarks: see the table above (`results/{scanrefer,multi3dref,scan2cap,sqa3d_test}/`).

**ScanQA val** (4675 q; Robin3D's scorer, plus EM with `answer_match` on `clean_answer`):
CIDEr 94.76, BLEU-4 12.92, METEOR 19.21, ROUGE-L 44.10, **EM@1 23.06, EM-R 41.52**
(`PREFIX_CACHE=1`: CIDEr 94.51, EM 22.93, EM-R 41.63).

**Real-3DQA** (official scorers; prompt = Robin3D's SQA3D format `<situation> <question> Answer
the question using a word or a phrase.`; the scene input is the same for every rotation):

| split | EM | EM-R |
|---|---|---|
| debiased_test (1485) | **26.94** | **31.92** |
| rotation 0 / 90 / 180 / 270 (750 each) | 12.40 / 9.47 / 11.87 / 11.20 | 17.20 / 12.93 / 16.13 / 15.20 |

Rotation consistency: One 48.93, Two 9.47, Three 2.27, Four 0.80, **VRS 15.37**
(`PREFIX_CACHE=1`: EM 26.80, EM-R 31.78, VRS 15.27).

**Beacon3D ScanNet** (1611 grounding / 1155 QA; official evaluators; QA judge
`gpt-4o-mini-2024-07-18`). Robin3D grounds to its own 150 Mask3D proposals; the chosen
proposal's box is mapped to the GT object of highest 3D box IoU (any overlap; the `_iou25`
variant also needs IoU >= 0.25). 2 of 1611 answers had no `<OBJxxx>` (counted wrong).

| | case | object | chain |
|---|---|---|---|
| grounding (any overlap) | **62.73** | **45.97** | 55.93 |
| grounding (IoU >= 0.25) | 60.50 | 43.90 | 53.94 |
| QA (GPT score) | **55.00** | **14.29** | |
| `PREFIX_CACHE=1`: grounding / QA | 62.67 / 54.76 | 45.97 / 14.03 | 55.84 |

QA per type: class 37.16, appearance 56.23, geometry 57.28, spatial 50.74, existence 65.10,
functionality 66.67; EM 49.44, EM-R 54.11; R1 36.36, R2 50.95. (The judge is re-run on every
scoring, so the per-type numbers move by a few tenths between scorings of the same predictions.)

## Faster evaluation: `PREFIX_CACHE=1`

Chat-Scene's fast path encodes each scene's prefix (system prompt + objects) once and lets all
questions of the scene reuse it. **That is not possible for Robin3D**: its prompt puts the
question *before* the 150 object tokens, so every object token already attends to the question and no KV is shared
across questions. What upstream wastes is per question: `generate(num_beams=5)` copies the
prompt (system prompt, question, 150 objects × 4 tokens) into each beam, re-encodes it, and every step `cat`s and beam-reorders
that cache. With `PREFIX_CACHE=1` the run script sets two config switches (both off by
default, which keeps the upstream behaviour):

- `eval_prompt_cache`: `Chat3D.generate_with_prompt_cache` runs the prompt except its last
  token through the base LLaMA once per question (right-padded batch, no logits) and starts
  `generate()` from the last prompt token without a cache. Every attention layer reads the
  question's prompt KV in place through `flash_attn_with_kvcache(cache_seqlens=len-1,
  cache_batch_idx=row // 5)`, one copy for its 5 beams, attends causally to its own
  generated keys, and merges the two by log-sum-exp
  (`LlamaFlashAttention2._shared_prefix_attention` in `models/modeling_llama.py`). Position
  ids get the per-row prompt length as offset (`prepare_inputs_for_generation`).
- `eval_gen_batch` (`GEN_BATCH`, default 32): this many questions per batched beam search.
  `BATCH_SIZE` defaults to 32 with it (8 without).

```bash
PREFIX_CACHE=1 bash extra_eval/run_extra_eval.sh          # GEN_BATCH=32 BATCH_SIZE=32
```

Full runs on 2 RTX PRO 6000 (`outputs/extra_eval_{upstream,prefixcache}_<ts>/`, same 12 val sets):

| | upstream | `PREFIX_CACHE=1` |
|---|---|---|
| inference, all 7 benchmarks (train.py's timer) | 2 h 38 min | **37 min (4.3×)** |
| whole job incl. model load + scoring | 2 h 48 min | 47 min |

Per set with `PREFIX_CACHE=1`: ScanRefer 8.1 min, Multi3DRefer 15, Scan2Cap 2.0, ScanQA 3.5,
SQA3D 2.9, Real-3DQA 3.2 (5 splits), Beacon3D grounding 1.3, QA 0.9. Scores: see the tables
above; all within 0.5 of upstream except Scan2Cap CIDEr@0.5 (-0.7).

`extra_eval/check_prompt_cache.py` compares the modes on a few scenes of one val set (one GPU;
`CHECK_SET`, `CHECK_SCENES`, `CHECK_MODES`, e.g. `"orig cache1 cache16"`), with time and peak
memory. ScanQA, 2 scenes, 134 questions: upstream 60.1 s / 20.4 GB, cache with 1 question per
search 35.5 s / 15.9 GB (97.0 % identical), 8 per search 13.0 s / 18.8 GB (94.0 % identical);
the differing answers are beam-search near-ties.

## Dependencies: `deps/`

`deps/` (gitignored) holds this project's own copy of every external dependency, and
`env.sh` points only there, so that deleting a shared cache or another project's data does
not break the pipeline. Files that already exist elsewhere on the same filesystem are hard
links (`cp -al`; no space, and they survive deletion of the other path); a symlink is used
only for a directory this project owns. A uv venv's `pyvenv.cfg` `home` points at uv's
managed Python, so a hard-linked copy of that interpreter (`deps/uv-python-3.10.22`) is kept
too and the venvs point at it.

| `deps/` | what | `env.sh` |
|---|---|---|
| `drive/` | authors' Google Drive: ckpt, ScanNet features, annotations (`download.sh`) | `ROBIN3D_DATA` |
| `venv/`, `venv-feats/`, `mask3d-venv/`, `uv-python-3.10.22/` | Python envs | `ROBIN3D_ENV`, `FEATS_ENV`, `MASK3D_ENV` |
| `vicuna-7b-v1.5/` | HF `lmsys/vicuna-7b-v1.5` | `LLAMA` |
| `jre11/` | Java for pycocoevalcap | `JAVA_HOME` |
| `real3dqa/` | HF `Oliver-Ma/Real-3DQA` | `REAL3DQA_DIR` |
| `benchmarks/scan2cap_val_corpus.json`, `scanrefer_unique_multiple_lookup.json` | HF `ZzZZCHS/Chat-Scene` `annotations/` (what Robin3D's scorers read on the authors' disk) | `SCAN2CAP_CORPUS`, `SCANREFER_LOOKUP` |
| `mask3d-src/`, `mask3d-ckpt/`, `scannet200/`, `uni3d-g/`, `dinov2-giant/` | feature rebuild models | `MASK3D_*`, `UNI3D_CKPT`, `DINOV2` |
| `3rscan_raw/`, `multiscan_raw/`, `multiscan_extracted/`, `rscan_scan_data/`, `multiscan_scan_data/` | raw scans + SceneVerse for 3RScan / MultiScan | `RSCAN_RAW`, `MSCAN_RAW`, `*_SCAN_DATA` |
| `scannet_scans/`, `scannet_raw/`, `feats/` | ScanNet feature check, rebuilt features | `SCANNET_*`, `FEATS_WORK` |

## Beacon3D 3RScan / MultiScan (rebuilt features)

Robin3D's inputs per object are three features of a Mask3D proposal, and the authors only
released them for ScanNet. `extra_eval/feats/` rebuilds them from the raw 3RScan / MultiScan
scans (mesh, superpoints, RGB-D frames, camera poses), following the released pipeline:

1. **Mask3D** (`scannet200_val.ckpt`, the official ScanNet200 test setting: 150 queries,
   top-750, DBSCAN 0.95), cut to the 150 best instances; each keeps the query feature
   (`decoder_norm` output) and query positional embedding of the query it came from
   (`mask3d_infer.py`, forward hooks, Mask3D unmodified).
2. **Uni3D-g** over each proposal's points (`uni3d.py`, same encoder as Chat-Scene).
3. **DINOv2-giant multi-view features** with the rule of Chat-Scene's `scannet_videofeat.py`
   (project the points into each frame with a depth check, convex hull of the proposal's
   visible points, mean patch token in its box, area-weighted over frames). 3RScan / MultiScan
   frames are rotated upright (from the camera pose) before DINOv2, since ScanNet's are.

Scenes are put in SceneVerse's axis-aligned frame (the Beacon3D object ids):
- 3RScan: exact vertex match with the raw mesh; official superpoints; every 5th `sequence.zip` frame.
- MultiScan: the raw mesh shifted; ScanNet Segmentator superpoints on the mesh; every 40th
  video frame, with ARKit poses mapped into the mesh frame.
- Poses were checked by projecting coloured vertices into the images.
A predicted proposal is mapped to the GT object of highest box IoU, as for ScanNet.

```bash
bash extra_eval/feats/run_prepare.sh       # CPU: scene packages (SCANNET_CHECK=1 adds the 16 ScanNet scenes)
bash extra_eval/feats/run_feats.sh         # GPU: Mask3D + Uni3D + DINOv2 (DOMAINS="scannet 3rscan multiscan")
VAL_TAG="beacon3d_3rscan_grounding#beacon3d_3rscan_qa#beacon3d_multiscan_grounding#beacon3d_multiscan_qa" \
  bash extra_eval/run_extra_eval.sh        # with FEATS_WORK set
PREDS_DIR=outputs/<ts> BENCHES=beacon3d_3rscan,beacon3d_multiscan bash extra_eval/score.sh
```

**Check on ScanNet.** The same pipeline was run on the 16 Beacon3D ScanNet scenes and
compared with the released files (`results/feature_check/`). Proposals were paired by box IoU,
and the cosines below are over the paired proposals.

| Mask3D input | paired box IoU | feat / pos / Uni3D / DINOv2 cosine |
|---|---|---|
| raw ScanNet coordinates (as the authors ran it) | 0.97 (142/150 >= 0.9) | 0.954 / 0.978 / 0.994 / 0.999 |
| axis-aligned coordinates (as 3RScan / MultiScan get) | 0.68 (69/150 >= 0.9) | 0.729 / 0.748 / 0.970 / 0.998 |

Robin3D on Beacon3D ScanNet with these rebuilt features stays close to the released-feature result:

| ScanNet features | grounding case / obj | QA case / obj |
|---|---|---|
| released | 62.73 / 45.97 | 55.00 / 14.29 |
| rebuilt, Mask3D on raw coordinates | 62.80 / 45.45 | 55.48 / 13.77 |
| rebuilt, Mask3D on axis-aligned coordinates | 61.90 / 43.38 | 54.83 / 14.29 |

**Results** (grounding with any-overlap mapping; IoU >= 0.25 in brackets; QA judge gpt-4o-mini):

| Beacon3D | grounding case / obj | QA case / obj | QA EM / EM-R |
|---|---|---|---|
| 3RScan (9 scans, 1267 / 816) | **37.04 / 23.90** (34.51 / 22.79) | **53.52 / 13.97** | 49.75 / 53.68 |
| MultiScan (5 scans, 657 / 477) | **40.33 / 26.42** (35.46 / 22.64) | **37.74 / 5.03** | 34.17 / 37.53 |

`<OBJxxx>` was missing in 1 / 4 grounding answers (counted wrong). 2D coverage is lower on
3RScan (100-144 of 150 proposals seen in a frame; short Tango sequences) than on ScanNet / MultiScan (~145).

## XR-QA / XR-SceneCaption (LSceneLLM, `extra_eval/xr/`)

LSceneLLM's (arXiv 2412.01292) large-scene benchmarks: XR-QA (val 2994 questions, 2981 unique
keys) and XR-SceneCaption (val 283), on scenes made of several SceneVerse HM3D regions.
Zero-shot = the released checkpoint; finetune = that checkpoint trained on the XR train splits
only. Objects are **Mask3D proposals** (no ground truth; the GT object count is used only for
the <=100 / >100 split of the scores).

**Inputs.** Robin3D's three per-object features are rebuilt for every region as for 3RScan /
MultiScan (*Beacon3D 3RScan / MultiScan* above): Mask3D `scannet200_val.ckpt` query feature +
positional embedding (`mask3d_xr.py infer`), Uni3D-g (`uni3d_xr.py`), and DINOv2-giant
multi-view features from views rendered from the HM3D v0.2 glb (`videofeats_xr.py`; SceneVerse
HM3D has no images; region -> glb offsets in `data/region_offsets.json`). An XR scene pools its
regions' proposals by score and keeps the 150 best (Robin3D's 150 object identifiers);
`mask3d_xr.py scenes` recomputes each query's positional embedding in the scene frame (the
range of all its regions), what Mask3D would give if the scene were one scan (`_rpos` sets keep
the per-region ones: 64.79 vs 64.64 QA CIDEr, no difference). Every scene got 150 proposals.

**Prompts.** XR-QA `short` = question + Robin3D's ScanQA suffix " Answer the question using a
word or a phrase." (`raw` = bare question); XR-SceneCaption = the XR question without its
`### human:` / `### assistant:` wrapper. Train items: one per reference answer (17,881 QA +
1,559 caption = 19,440).

**Finetune.** Robin3D's stage-2 recipe (`scripts/run_stage2.sh`: LoRA r16, lr 1e-5, cosine,
warmup 0.1, max_grad_norm 0.01, batch 8 per GPU, seed 42) for 3 epochs, 1215 steps per epoch on
2 GPUs; val after every epoch. Resumable after a Slurm requeue (it was preempted twice and
continued from `ckpt_<k>.pth`).

**Scoring** (`score.py`): LSceneLLM's capeval (CIDEr / BLEU-4 / METEOR / ROUGE-L x100) on
`raw` text, and on Robin3D's own ScanQA normalisation (`robin3d`: cut at the first `<OBJxxx>`,
strip the trailing '.', lower-case the first letter, PTB tokenizer; no `clean_answer`).

| Setting | Task | norm | CIDEr | BLEU-4 | METEOR | ROUGE-L | <=100 obj C | >100 obj C | words |
|---|---|---|---|---|---|---|---|---|---|
| zero-shot | XR-QA short | robin3d | **64.64** | 0.00 | 8.73 | 26.53 | 65.92 | 62.32 | 2.4 |
| zero-shot | XR-QA raw | robin3d | 61.34 | 1.29 | 8.60 | 25.24 | 62.23 | 59.83 | 3.0 |
| zero-shot | XR-SceneCaption | robin3d | **0.03** | 0.01 | 2.02 | 8.85 | 0.01 | 0.05 | 14.0 |
| finetune ep 3 | XR-QA short | robin3d | **159.26** | 20.02 | 19.55 | 47.85 | 160.31 | 156.97 | 4.0 |
| finetune ep 3 | XR-QA short | raw | 128.33 | 17.57 | 20.52 | 40.05 | 128.33 | 127.80 | 4.0 |
| finetune ep 3 | XR-SceneCaption | robin3d | **4.33** | 4.13 | 12.49 | 19.21 | 3.99 | 4.68 | 80.6 |
| finetune ep 3 | XR-SceneCaption | raw | 3.53 | 3.63 | 13.68 | 17.84 | 3.41 | 3.72 | 80.6 |

Per epoch (robin3d norm, QA / caption CIDEr): 156.62 / 5.67, 159.51 / 4.54, 159.26 / 4.33.
Zero-shot raw-text QA CIDEr is only 17.35 (answers like "white" vs references like "The
cabinet is white."); after finetuning the answers take the references' phrasing. One val
caption scene (of 283 items: 1) also appears in XR-SceneCaption train; that is LSceneLLM's split.

```bash
bash extra_eval/xr/run_xr_feats.sh       # GPU: Mask3D + Uni3D + DINOv2 for all XR regions (56 min on 2 GPUs)
bash extra_eval/xr/run_xr_eval.sh        # zero-shot + score (40 min)
bash extra_eval/xr/run_xr_finetune.sh    # 3 epochs + val each epoch + score (~1 h per epoch)
OUT=outputs/xr_eval_<ts> bash extra_eval/xr/score.sh   # re-score an output dir only (CPU)
```

Results: `results/xr/{zeroshot,finetune}/` (predictions, `*_scores.json`, `scores.json`,
`config.json`; no checkpoints). Upstream changes for XR: `dataset/dataset_val.py` keys its
feature cache by all feature files (upstream: Uni3D + 2D file only, so the `_rpos` / `_no2d`
sets silently reused the first set's features); `tasks/train.py` leaves `xrqa_*` / `xrcap_*`
to `score.py`.

## Aligned scoring (`extra_eval/aligned/`)

A second scoring of the saved predictions with one text normalisation shared by all baselines
of this project, so the numbers are comparable across baselines (`aligned_eval.py` is the shared
scorer). The scores above (paper / each benchmark's own protocol) stay as they are; these are extra.

- Prediction cleanup: only the model's special tokens are removed (`<OBJxxx>`, `<s>`, `</s>`,
  `<unk>`); no truncation.
- Scan2Cap: PTB tokenizer + `sos ... eos` on both sides, untruncated references
  (`deps/aligned/scan2cap_val_corpus.json`, the `SCAN2CAP_CORPUS` file), Robin3D's own
  proposal matching (box IoU >= 0.25 / 0.5).
- ScanQA, XR-QA: `clean_answer` on both sides, then PTB, for CIDEr / BLEU / METEOR / ROUGE-L;
  EM / EM-R = `clean_answer` on both sides, exact match over all references before the
  substring (refined) match. SQA3D: EM / EM-R the same way.
- XR-SceneCaption: PTB on both sides (no `clean_answer`, no sos/eos). XR also keeps a `raw` block
  (LSceneLLM protocol, no PTB) and is scored per item (2994 XR-QA / 283 XR-SceneCaption), with no
  episode dedup.
- Real-3DQA and Beacon3D QA: unchanged (not re-scored).

```bash
bash extra_eval/aligned/run_score_aligned.sh                  # all; --only scan2cap,scanqa,sqa3d,xrcap,xrqa
```

CPU only; needs Java (`deps/jre11` unless `JAVA_HOME` is set) and `ROBIN3D_DATA` (Scan2Cap boxes).
Scan2Cap / ScanQA / SQA3D read the prediction dir recorded in `results/summary.json`
(`pred_dir`, i.e. a local `run_extra_eval.sh` output scored by `score.sh`); XR reads the
committed `results/xr/` predictions. Output: `results/aligned/<bench>__<variant>.json` and
`results/aligned/summary.json` (XR headline = `short` prompt zero-shot, last finetune epoch).

| | aligned | own protocol (above) |
|---|---|---|
| Scan2Cap CIDEr@0.5 | 86.60 | 86.60 (authors' log 87.22) |
| ScanQA CIDEr / EM / EM-R | 94.77 / 23.06 / 41.52 | 94.76 / 23.06 / 41.52 |
| SQA3D test EM / EM-R | 56.86 / 59.73 | 56.86 / 59.73 |
| XR-QA zero-shot CIDEr (clean_ptb) / EM / EM-R | 78.65 / 15.16 / 22.24 | 64.64 (robin3d norm) / – / – |
| XR-QA finetune ep 3 CIDEr (clean_ptb) / EM / EM-R | 141.43 / 25.42 / 30.79 | 159.26 (robin3d norm) / – / – |
| XR-SceneCaption zero-shot / finetune ep 3 CIDEr (ptb) | 0.03 / 4.33 | 0.03 / 4.33 |

Robin3D's own scorers already match the aligned protocol on Scan2Cap, ScanQA and SQA3D. On
XR-QA the robin3d norm has no `clean_answer` and dedups by episode (2981 keys), hence the gap.


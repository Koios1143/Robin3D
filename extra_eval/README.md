# Extra benchmarks for Robin3D

Evaluation of the **released Robin3D checkpoint** (`ckpt_for_robin3d_final_performance/ckpt_01.pth`
from the authors' Google Drive) on benchmarks the paper does not report: **ScanQA** (with EM),
**Real-3DQA** and **Beacon3D** (ScanNet with the released features; 3RScan / MultiScan with
features rebuilt the same way). No training; the model, ScanNet features and prompts are the
authors' own.

## Run

```bash
cp env.sh.example env.sh                       # fill in paths (+ OPENAI_API_KEY for Beacon3D QA)
bash extra_eval/build_env_cu128.sh             # venv: py3.10, torch 2.7.1+cu128 (Blackwell OK)
bash extra_eval/download.sh                    # ckpt + ScanNet Mask3D/Uni3D/DINOv2 features + val annos
bash extra_eval/run_extra_eval.sh              # inference, all visible GPUs (~1.5 h on 2 RTX PRO 6000)
PREDS_DIR=outputs/extra_eval_<ts> bash extra_eval/score.sh   # CPU; Java for pycocoevalcap
```

Google Drive sometimes refuses a file for a while ("too many accesses"); `download.sh` skips
finished files, so just run it again later.

Changes to upstream code (all needed only to run evaluation outside the authors' machine):
`dataset/base_dataset.py` reads the Vicuna tokenizer path from `$LLAMA`; `tasks/shared_utils.py`
loads the checkpoint with `weights_only=False` (it pickles the config; torch >= 2.6);
`tasks/train.py` skips its online scorer for the new sets, which `extra_eval/score.py` scores.

## Setup check

Robin3D's `OBJ_lm_head` (the logits of the `<OBJxxx>` tokens) is frozen and is not in the
checkpoint, so it comes from the seeded random init (seed 42, rank 0 broadcast by DDP). To make
sure this run rebuilt the same model, ScanRefer val and SQA3D test are re-run and compared with
the authors' released predictions (`preds_epoch1_step5146_*.json`, the run that produced `ckpt_01`):

| | ours | authors' log | identical generations |
|---|---|---|---|
| ScanRefer Acc@0.25 / @0.5 | 60.76 / 55.01 | 60.81 / 55.07 | 99.5 % |
| SQA3D test EM / EM-R | 56.86 / 59.73 | 56.89 / 59.76 | 98.8 % |
| ScanQA CIDEr / B-4 | 94.76 / 12.92 | 94.80 / 12.67 | 96.4 % |

## Results (`results/`)

**ScanQA val** (4675 q; Robin3D's scorer, plus EM with `answer_match` on `clean_answer`):
CIDEr 94.76, BLEU-4 12.92, METEOR 19.21, ROUGE-L 44.10, **EM@1 23.06, EM-R 41.52**.

**Real-3DQA** (official scorers; prompt = Robin3D's SQA3D format `<situation> <question> Answer
the question using a word or a phrase.`; the scene input is the same for every rotation):

| split | EM | EM-R |
|---|---|---|
| debiased_test (1485) | **26.94** | **31.92** |
| rotation 0 / 90 / 180 / 270 (750 each) | 12.40 / 9.47 / 11.87 / 11.20 | 17.20 / 12.93 / 16.13 / 15.20 |

Rotation consistency: One 48.93, Two 9.47, Three 2.27, Four 0.80, **VRS 15.37**.

**Beacon3D ScanNet** (1611 grounding / 1155 QA; official evaluators; QA judge
`gpt-4o-mini-2024-07-18`). Robin3D grounds to its own 150 Mask3D proposals; the chosen
proposal's box is mapped to the GT object of highest 3D box IoU (any overlap; the `_iou25`
variant also needs IoU >= 0.25). 2 of 1611 answers had no `<OBJxxx>` (counted wrong).

| | case | object | chain |
|---|---|---|---|
| grounding (any overlap) | **62.73** | **45.97** | 55.93 |
| grounding (IoU >= 0.25) | 60.50 | 43.90 | 53.94 |
| QA (GPT score) | **55.00** | **14.29** | |

QA per type: class 36.94, appearance 56.38, geometry 57.44, spatial 50.66, existence 65.00,
functionality 66.67; EM 49.44, EM-R 54.11; R1 36.59, R2 51.18.

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

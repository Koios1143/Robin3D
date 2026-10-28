"""Check and time the shared prompt KV (config eval_prompt_cache / eval_gen_batch).

Runs one val set, restricted to its first CHECK_SCENES scenes (default 4), once per mode in
CHECK_MODES (default "orig cache1 cache16") with the same model:
  orig    upstream: one generate() per question, its prompt copied to and encoded for each beam
  cacheN  each prompt encoded once and read in place by its 5 beams, N questions per batched
          beam search
Same inputs in every mode; only the attention split, padding and batch shape differ, so the
generations should agree up to bf16 near-ties. Prints every mode's agreement with the first
mode, its time and peak GPU memory. Same arguments as tasks/train.py (config file +
overrides); launch with torchrun (1 GPU):
  CHECK_SET=scanqa torchrun --nproc_per_node=1 extra_eval/check_prompt_cache.py <config> <overrides>
Writes <output_dir>/check_<set>.json.
"""
import json
import os
import re
import time

import torch
from torch.utils.data import DataLoader, Subset

from dataset.dataset_val import val_collate_fn
from models.chat3d import Chat3D
from tasks.shared_utils import setup_model
from tasks.train import setup_dataloaders
from utils.basic_utils import setup_seed
from utils.config_utils import setup_main


def run(model, loader, device):
    preds = []
    torch.cuda.synchronize()
    t0 = time.time()
    for batch in loader:
        for k in batch:
            if type(batch[k]) == torch.Tensor:
                batch[k] = batch[k].to(device)
        with torch.no_grad():
            preds += model(batch)
    torch.cuda.synchronize()
    return preds, time.time() - t0


def main(config):
    name = config.val_tag
    n_scenes = int(os.environ.get("CHECK_SCENES", 4))
    # the same order of seeded steps as tasks/train.py main(): OBJ_lm_head is not in the
    # checkpoint, so its random init has to come out the same
    setup_seed(config.seed)
    device = torch.device(config.device)
    _, val_loaders = setup_dataloaders(config)
    config.scheduler.num_training_steps = config.scheduler.num_warmup_steps = 0  # as main() with no train set
    model, model_without_ddp, *_ = setup_model(config, model_cls=Chat3D, find_unused_parameters=False)
    model.eval()
    model_without_ddp.llama_model.config.use_cache = True  # as evaluate_all()

    ds = val_loaders[0].dataset.datasets[0]
    scenes = sorted({a["scene_id"] for a in ds.anno})[:n_scenes]
    idx = [i for i, a in enumerate(ds.anno) if a["scene_id"] in scenes]
    loader = DataLoader(Subset(ds, idx), batch_size=config.batch_size, num_workers=2, collate_fn=val_collate_fn)

    out = {"set": name, "scenes": scenes, "n": len(idx)}
    preds = {}
    modes = os.environ.get("CHECK_MODES", "orig cache1 cache16").split()
    for mode in modes:
        model_without_ddp.config.eval_prompt_cache = mode != "orig"
        model_without_ddp.config.eval_gen_batch = int(re.sub(r"\D", "", mode) or 1)
        torch.cuda.reset_peak_memory_stats()
        preds[mode], out[f"sec_{mode}"] = run(model, loader, device)
        out[f"gb_{mode}"] = torch.cuda.max_memory_allocated() / 2**30
        print(f"[{name}] {mode}: {out[f'sec_{mode}']:.1f}s for {len(idx)} questions, "
              f"peak {out[f'gb_{mode}']:.1f} GB", flush=True)

    ref = modes[0]
    for mode in modes[1:]:
        out[f"agree_{mode}_{ref}"] = sum(x == y for x, y in zip(preds[mode], preds[ref])) / len(idx)
    out["diffs"] = {mode: [{"prompt": ds.anno[i]["prompt"] if "prompt" in ds.anno[i] else None, ref: r, mode: m}
                           for i, r, m in zip(idx, preds[ref], preds[mode]) if r != m][:20]
                    for mode in modes[1:]}
    print(json.dumps({k: v for k, v in out.items() if k != "diffs"}, indent=1))
    json.dump(out, open(os.path.join(config.output_dir, f"check_{name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main(setup_main())

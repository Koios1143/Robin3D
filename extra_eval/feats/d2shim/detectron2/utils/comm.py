"""Minimal stand-in for detectron2.utils.comm (Mask3D's training loss only)."""
import torch.distributed as dist


def get_world_size():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

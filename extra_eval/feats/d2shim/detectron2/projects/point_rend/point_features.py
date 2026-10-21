"""Minimal stand-in for detectron2's point_rend helpers, which Mask3D imports for its
training losses only (inference never calls them). ``point_sample`` follows detectron2."""
import torch.nn.functional as F


def point_sample(input, point_coords, **kwargs):
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def get_uncertain_point_coords_with_randomness(*args, **kwargs):
    raise NotImplementedError("training-only (detectron2 stand-in)")

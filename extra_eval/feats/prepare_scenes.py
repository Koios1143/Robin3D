#!/usr/bin/env python3
"""Raw scans -> one package per scene for the Robin3D feature rebuild (``build_scene_feats.py``).

Robin3D needs, per Mask3D proposal, Mask3D query features, Uni3D (point cloud) and DINOv2
(multi-view RGB) features. This script brings each scene into one axis-aligned frame:

* ``3rscan``: points, colours and instance ids from SceneVerse's
  ``pcd_with_global_alignment`` (the Beacon3D object ids). They are the raw mesh vertices
  of ``labels.instances.annotated.v2.ply`` rotated about z and shifted; that rigid transform
  is fitted per scan (residual ~1e-14 m) and applied to the camera poses of
  ``sequence.zip`` (every ``--rscan_skip``-th frame; Tango sequences are short).
  Superpoints = the scan's own ``mesh.refined.0.010000.segs.v2.json`` (ScanNet Segmentator
  output, same vertex order).
* ``multiscan``: all vertices of ``<scene>.ply`` (colours, normals); SceneVerse's cloud is
  that mesh shifted (subsampled for large scenes), so the shift is fitted by ICP (exact to
  ~1e-12 m, ~5 mm where SceneVerse resampled) and each vertex takes the instance id of the
  nearest SceneVerse point. Frames are decoded from the ``.mp4`` / ``.depth.zlib`` streams
  (every ``--mscan_skip``-th frame, colour resized to 640x480); the ARKit poses
  (column-major camera->world) are mapped into the mesh frame with the inverse of
  ``align.json``'s ``coordinate_transform`` and converted to the OpenCV camera convention.
  MultiScan ships no superpoints, so ScanNet Segmentator's rule (Felzenszwalb, edge weight
  1 - n1.n2, squared on convex edges, kThresh 0.01, segments < 20 vertices merged) is run
  on the mesh edges.
* ``scannet`` (only to validate the rebuild against Robin3D's released ScanNet files):
  ``<scene>_vh_clean_2.ply`` + ``axisAlignment``; frames from ``.sens`` exactly as
  OpenScene's ``preprocess_2d_scannet.py`` (every 20th frame, 640x480, nearest resize).

Raw inputs are only read. Per scene ``<out_dir>/<domain>/<alias>/``: ``points.npz``
(coords aligned, colors 0-255, inst = GT id or -1, seg), ``frames/<i>.jpg`` + ``<i>.depth.png``
(uint16 mm, colour resolution) + ``<i>.pose.txt`` (camera->aligned world, OpenCV axes) +
``<i>.K.txt``. Also ``<out_dir>/<domain>/scannet200/`` (Mask3D processed layout) and
``<out_dir>/<domain>/scene_map.json`` (alias -> raw scene id). Aliases look like
``rscan0000_00`` because Robin3D recovers the scene of a feature key as
``'_'.join(key.split('_')[:2])``.
"""

import argparse
import io
import sys
import json
import os
import shutil
import zipfile
import zlib
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PREFIX = {"3rscan": "rscan", "multiscan": "mscan"}
SEG_THR, SEG_MIN = 0.01, 20


def fit_rigid(src, dst):
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    ms, md = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - ms).T @ (dst - md))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    t = md - r @ ms
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = r, t
    return T, float(np.abs(src @ r.T + t - dst).max())


def fit_translation(raw, sv):
    """t such that sv is (a subset / resampling of) raw + t; ICP with the rotation fixed."""
    sub = sv[np.random.default_rng(0).choice(len(sv), min(50000, len(sv)), replace=False)]
    tree = cKDTree(raw)
    t = np.r_[(sv[:, :2].min(0) + sv[:, :2].max(0) - raw[:, :2].min(0) - raw[:, :2].max(0)) / 2,
              sv[:, 2].min() - raw[:, 2].min()]
    for _ in range(100):
        j = tree.query(sub - t)[1]
        t2 = (sub - raw[j]).mean(0)
        if np.abs(t2 - t).max() < 1e-12:
            break
        t = t2
    d = tree.query(sv - t)[0]
    return t, float(np.median(d)), float(d.max())


def segmentator(xyz, normals, edges):
    """ScanNet Segmentator (graph segmentation + small-segment merge) on the given edges."""
    a, b = edges[:, 0], edges[:, 1]
    nrm = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    d = xyz[b] - xyz[a]
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    w = 1.0 - (nrm[a] * nrm[b]).sum(1)
    w = np.where((nrm[b] * d).sum(1) > 0, w * w, w)
    order = np.argsort(w, kind="stable")
    a, b, w = a[order].tolist(), b[order].tolist(), w[order].tolist()
    n = len(xyz)
    parent, rank, size, thr = list(range(n)), [0] * n, [1] * n, [SEG_THR] * n

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def join(x, y):
        if rank[x] > rank[y]:
            parent[y] = x
            size[x] += size[y]
            return x
        parent[x] = y
        size[y] += size[x]
        if rank[x] == rank[y]:
            rank[y] += 1
        return y

    for i in range(len(w)):
        x, y = find(a[i]), find(b[i])
        if x != y and w[i] <= thr[x] and w[i] <= thr[y]:
            r = join(x, y)
            thr[r] = w[i] + SEG_THR / size[r]
    for i in range(len(w)):
        x, y = find(a[i]), find(b[i])
        if x != y and (size[x] < SEG_MIN or size[y] < SEG_MIN):
            join(x, y)
    return np.array([find(i) for i in range(n)])


def write_frame(fdir, i, color_bgr, depth_m, pose, K):
    import cv2
    cv2.imwrite(str(fdir / f"{i}.jpg"), color_bgr)
    dep = np.clip(np.nan_to_num(depth_m) * 1000.0, 0, 65535).astype(np.uint16)
    if dep.shape[:2] != color_bgr.shape[:2]:
        dep = cv2.resize(dep, (color_bgr.shape[1], color_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(fdir / f"{i}.depth.png"), dep)
    np.savetxt(fdir / f"{i}.pose.txt", pose)
    np.savetxt(fdir / f"{i}.K.txt", K)


def sceneverse(root, scene):
    xyz, rgb, inst = torch.load(Path(root) / "pcd_with_global_alignment" / f"{scene}.pth", weights_only=False)
    return np.asarray(xyz, np.float32), np.asarray(rgb, np.float32), np.asarray(inst).astype(np.int64)


def prep_3rscan(scene, args, pdir):
    from plyfile import PlyData
    import cv2
    raw = Path(args.rscan_raw) / "scans" / scene
    xyz, rgb, inst = sceneverse(args.rscan_sceneverse, scene)
    v = PlyData.read(str(raw / "labels.instances.annotated.v2.ply"))["vertex"]
    T, res = fit_rigid(np.stack([v["x"], v["y"], v["z"]], 1), xyz)
    assert res < 1e-4 and (np.asarray(v["objectId"]) == inst).all(), (scene, res)
    seg = np.asarray(json.load(open(raw / "mesh.refined.0.010000.segs.v2.json"))["segIndices"])
    fdir = pdir / "frames"
    fdir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(raw / "sequence.zip") as z:
        info = dict(l.split(" = ") for l in z.read("_info.txt").decode().strip().splitlines())
        K = np.array(info["m_calibrationColorIntrinsic"].split(), float).reshape(4, 4)[:3, :3]
        shift = float(info["m_depthShift"])
        n = int(info["m_frames.size"])
        for i in range(0, n, args.rscan_skip):
            name = f"frame-{i:06d}"
            color = cv2.imdecode(np.frombuffer(z.read(f"{name}.color.jpg"), np.uint8), cv2.IMREAD_COLOR)
            depth = cv2.imdecode(np.frombuffer(z.read(f"{name}.depth.pgm"), np.uint8), cv2.IMREAD_UNCHANGED)
            pose = np.loadtxt(io.BytesIO(z.read(f"{name}.pose.txt")))
            if not np.isfinite(pose).all():
                continue
            write_frame(fdir, i, color, depth.astype(np.float32) / shift, T @ pose, K)
    return xyz, rgb, inst, seg, np.zeros_like(xyz), T, res


def prep_multiscan(scene, args, pdir):
    from plyfile import PlyData
    import cv2
    raw = Path(args.mscan_raw) / scene
    sv_xyz, _, sv_inst = sceneverse(args.mscan_sceneverse, scene)
    ply = PlyData.read(str(raw / f"{scene}.ply"))
    v = ply["vertex"]
    P = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    # SceneVerse = the mesh vertices shifted (some scenes subsampled to 240k points)
    t, res, res_max = fit_translation(P, sv_xyz.astype(np.float64))
    assert res_max < 0.05, (scene, res, res_max)
    T = np.eye(4)
    T[:3, 3] = t
    xyz = (P + t).astype(np.float32)
    rgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.float32)
    inst = sv_inst[cKDTree(sv_xyz).query(xyz)[1]]  # Beacon3D ids = SceneVerse instance ids
    normals = np.stack([v["nx"], v["ny"], v["nz"]], 1)
    faces = np.stack(ply["face"]["vertex_indices"])
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.unique(np.sort(e, 1), axis=0)
    seg = segmentator(xyz.astype(np.float64), normals, e)
    fdir = pdir / "frames"
    fdir.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(raw / f"{scene}.json"))
    dstream = next(s for s in meta["streams"] if s["id"].startswith("depth"))
    dh, dw = dstream["resolution"]
    # one raw-deflate stream (no zlib header) of float16 metres
    depth_all = np.frombuffer(zlib.decompressobj(-15).decompress(open(raw / f"{scene}.depth.zlib", "rb").read()),
                              np.float16)
    depth_all = depth_all.reshape(-1, dh, dw)
    cams = [json.loads(l) for l in open(raw / f"{scene}.jsonl")]
    flip = np.diag([1.0, -1.0, -1.0, 1.0])  # ARKit camera (y up, z back) -> OpenCV (y down, z forward)
    # the mesh lives in inv(align) of the ARKit world (checked: depth / colour agreement of
    # projected vertices, against the other conventions)
    A = np.asarray(json.load(open(raw / f"{scene}.align.json"))["coordinate_transform"], float).reshape(4, 4).T
    to_mesh = np.linalg.inv(A)
    cap = cv2.VideoCapture(str(raw / f"{scene}.mp4"))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % args.mscan_skip == 0 and i < len(cams) and i < len(depth_all):
            c = cams[i]
            pose = to_mesh @ np.asarray(c["transform"], float).reshape(4, 4).T @ flip  # column-major
            K = np.asarray(c["intrinsics"], float).reshape(3, 3).T
            h, w = frame.shape[:2]
            sx, sy = 640.0 / w, 480.0 / h
            K = np.diag([sx, sy, 1.0]) @ K
            write_frame(fdir, i, cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA),
                        depth_all[i].astype(np.float32), T @ pose, K)
        i += 1
    cap.release()
    assert i == len(cams), (scene, i, len(cams))
    return xyz, rgb, inst, seg, normals, T, res


def prep_scannet(scene, args, pdir):
    import cv2
    from plyfile import PlyData
    d = Path(args.scannet_scans) / scene
    v = PlyData.read(str(d / f"{scene}_vh_clean_2.ply"))["vertex"]
    coords = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    rgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.float32)
    A = np.eye(4)
    for line in open(d / f"{scene}.txt"):
        if line.startswith("axisAlignment"):
            A = np.array(line.split()[-16:], float).reshape(4, 4)
    xyz = (coords @ A[:3, :3].T + A[:3, 3]).astype(np.float32)
    seg = np.asarray(json.load(open(d / f"{scene}_vh_clean_2.0.010000.segs.json"))["segIndices"])
    from scannet_sensordata import SensorData
    sd = SensorData(str(Path(args.scannet_sens) / scene / f"{scene}.sens"))
    fdir = pdir / "frames"
    fdir.mkdir(parents=True, exist_ok=True)
    K = np.array([[577.870605, 0, 319.5], [0, 577.870605, 239.5], [0, 0, 1]])  # OpenScene's fixed intrinsics
    for i in range(0, len(sd.frames), 20):
        f = sd.frames[i]
        color = cv2.resize(f.decompress_color(sd.color_compression_type), (640, 480), interpolation=cv2.INTER_NEAREST)
        depth = np.frombuffer(f.decompress_depth(sd.depth_compression_type), np.uint16).reshape(sd.depth_height, sd.depth_width)
        depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_NEAREST)
        write_frame(fdir, i, color[:, :, ::-1].copy(), depth.astype(np.float32) / 1000.0, A @ f.camera_to_world, K)
    return xyz, rgb, -np.ones(len(xyz), np.int64), seg, np.zeros_like(xyz), A, 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--domain", required=True, choices=["3rscan", "multiscan", "scannet"])
    ap.add_argument("--scenes", required=True, help="json list of raw scene ids")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mask3d_meta", required=True, help="dir with Mask3D scannet200 label_database.yaml + color_mean_std.yaml")
    ap.add_argument("--rscan_raw", default="")
    ap.add_argument("--rscan_sceneverse", default="", help="SceneVerse 3RScan scan_data")
    ap.add_argument("--mscan_raw", default="", help="dir with <scene>/<scene>.{ply,json,jsonl,mp4,depth.zlib} (unzipped MultiScan scenes)")
    ap.add_argument("--mscan_sceneverse", default="", help="SceneVerse MultiScan scan_data")
    ap.add_argument("--scannet_scans", default="", help="dir with <scene>/<scene>_vh_clean_2.ply, .txt, segs")
    ap.add_argument("--scannet_sens", default="", help="dir with <scene>/<scene>.sens")
    ap.add_argument("--rscan_skip", type=int, default=5)
    ap.add_argument("--mscan_skip", type=int, default=40)
    args = ap.parse_args()
    scenes = json.load(open(args.scenes))
    out = Path(args.out_dir) / args.domain
    m3d = out / "scannet200"
    (m3d / "validation").mkdir(parents=True, exist_ok=True)
    for f in ["label_database.yaml", "color_mean_std.yaml"]:
        shutil.copy(Path(args.mask3d_meta) / f, m3d / f)
    smap, db = {}, []
    for k, scene in enumerate(scenes):
        alias = scene if args.domain == "scannet" else f"{PREFIX[args.domain]}{k:04d}_00"
        pdir = out / alias
        prep = {"3rscan": prep_3rscan, "multiscan": prep_multiscan, "scannet": prep_scannet}[args.domain]
        if not (pdir / "points.npz").exists():
            pdir.mkdir(parents=True, exist_ok=True)
            xyz, rgb, inst, seg, normals, T, res = prep(scene, args, pdir)
            seg = np.unique(seg, return_inverse=True)[1]
            np.savez_compressed(pdir / "points.npz", coords=xyz, colors=rgb, inst=inst, seg=seg, raw_to_aligned=T)
            print(f"{scene} -> {alias}: {len(xyz)} points, {seg.max() + 1} superpoints, "
                  f"{len(list((pdir / 'frames').glob('*.jpg')))} frames, fit residual {res:.1e}", flush=True)
        p = np.load(pdir / "points.npz")
        n = len(p["coords"])
        npy = m3d / "validation" / f"{alias}.npy"
        if not npy.exists():
            # Mask3D processed row: xyz rgb normal(unused: add_normals=false) segment semantic
            # (0 = ignored) instance. Mask3D's collate drops scenes without instances, and the
            # target is also what carries the superpoints, so scenes without GT (the ScanNet
            # check) get their superpoint id as a stand-in instance; labels are never used.
            inst = p["inst"] if (p["inst"] > 0).any() else p["seg"] + 1
            pts = np.hstack([p["coords"], p["colors"], np.zeros((n, 3)), p["seg"][:, None],
                             np.zeros((n, 1)), inst[:, None]]).astype(np.float32)
            np.save(npy, pts)
        db.append({"filepath": os.path.abspath(m3d / "validation" / f"{alias}.npy"), "file_len": n,
                   "raw_filepath": f"{args.domain}/{alias}/{alias}.pth", "scene": alias})
        smap[alias] = scene
    with open(m3d / "validation_database.yaml", "w") as f:
        yaml.safe_dump(db, f)
    json.dump(smap, open(out / "scene_map.json", "w"), indent=1)


if __name__ == "__main__":
    main()

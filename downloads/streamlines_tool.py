# -*- coding: utf-8 -*-
"""
streamlines_tool.py
------------------------------------------------------------
What this script does:
- Reads a 3D binary volume (0=pore, 1=solid) and a 3D Cmap volume
- Builds a pseudo vector field using v = -grad(Cmap)
- Randomly seeds points in the pore space and integrates streamlines
- Renders a semi-transparent solid skeleton + blue speed-colored streamlines + comet-tail glow
- Exports an MP4 video (libx264)

You only need to edit the CONFIG block below.
------------------------------------------------------------
"""

import argparse
import os
from pathlib import Path
import numpy as np


# ============================================================
# CONFIG (Edit ONLY here)
# ============================================================
ROOT = Path(__file__).resolve().parent

CONFIG = {
    # --------------------------
    # Input / Output files (most frequently edited)
    # --------------------------
    "BINARY_PATH": str(ROOT / "test1.tif"),                 # 3D binary: 0=pore, 1=solid
    "CMAP_PATH":   str(ROOT / "test1_pBd3_Cmap.tif"),       # 3D Cmap: .tif/.tiff or .mat (.mat needs scipy)
    "OUTPUT_MP4":  str(ROOT / "streamlines_tool.mp4"),      # output video filename

    # --------------------------
    # Voxel size (VERY IMPORTANT: affects aspect ratio / "boxy" look)
    # --------------------------
    "VOXEL_UM": 0.43,   # CT voxel size in micrometers (um/voxel). Wrong value => weird proportions.

    # --------------------------
    # Video quality / speed
    # --------------------------
    "DURATION": 10.0,   # seconds. Larger => longer video
    "FPS": 20,          # frames per second. 30 is common for smoother motion
    "WIDTH": 854,       # output width (pixels)
    "HEIGHT": 480,      # output height (pixels)
    "DPI": 130,         # higher => sharper but slower (160 is common if you want crisp frames)

    # --------------------------
    # Solid skeleton (surface mesh)
    # --------------------------
    "SOLID_ALPHA": 0.14,  # transparency of the gray skeleton (0.08~0.20)
    "DS_MESH": 4,         # mesh downsample: smaller => finer but slower (2 finer; 4 faster)
    "MESH_STEP": 2,       # marching cubes step: smaller => finer but slower (1 finer; 2 faster)

    # --------------------------
    # Vector field from Cmap: v = -grad(Cmap)
    # --------------------------
    "DS_FIELD": 2,        # field downsample: 1 = finest/slowest; 2 = common; 3 = faster but coarser
    "SMOOTH_SIGMA": 1.0,  # Gaussian smoothing on Cmap: if streamlines break too often, try 1.5~2.0

    # --------------------------
    # Streamline generation (most important for “looks”)
    # --------------------------
    "N_LINES": 200,       # number of random seeds. Larger => denser streamlines but slower (300 looks fuller)
    "SEED": 0,            # RNG seed. Keep fixed => reproducible; change => different streamline layout

    "STEP": 0.7,          # integration step size: too large => hits walls & breaks; too small => slower (0.5~1.0)
    "N_STEPS": 200,       # max steps per direction: larger => longer streamlines but slower (260 for longer)
    "MIN_POINTS": 30,     # keep only lines with >= this points. Want more lines? lower to 20.

    # --------------------------
    # Line appearance (if too faint / too thick, adjust here)
    # --------------------------
    "LINE_ALPHA": 0.7,    # streamline transparency (0.5~0.8)
    "LINE_WIDTH": 1.5,    # line width (1.5 or 2.0 for stronger visual)

    # --------------------------
    # Comet-tail glow (the blue “moving dots”)
    # --------------------------
    "ENABLE_GLOW": True,  # set False to remove the “blue balls”
    "GLOW_WINDOW": 12,    # tail length window: larger => longer tail (10~20)
    "GLOW_SPEED": 1.2,    # motion speed along the line: larger => faster (0.8~2.0)
    "GLOW_SIZE": 12.0,    # dot size: smaller if too big (10~16)

    # --------------------------
    # Camera motion (controls 3D feeling)
    # --------------------------
    "ELEV": 18.0,         # elevation angle: larger => more top-down (15~30)
    "AZIM0": -60.0,       # initial azimuth
    "AZIM_SPAN": 30.0,    # rotation span: larger => more rotation (30~60)
    "ZOOM_AMP": 0.10,     # breathing zoom: 0=no zoom; 0.1 subtle; 0.15 stronger

    # --------------------------
    # Crop (auto-crop around solid bbox to speed up & reduce empty space)
    # --------------------------
    "NO_CROP": False,     # True=disable cropping (use this if you suspect cropping causes shape issues)
}
# ============================================================


# -----------------------------
# Read 3D TIFF stack (multi-page TIFF -> (z,y,x))
# -----------------------------
def read_tiff_stack(path: str) -> np.ndarray:
    """Read a multi-page TIFF into a 3D array (z, y, x)."""
    try:
        import tifffile
        vol = tifffile.imread(path)
        if vol.ndim != 3:
            raise RuntimeError(f"TIFF is not 3D: {path}, shape={vol.shape}")
        return vol
    except Exception:
        # Fallback reader via PIL (slower but more compatible)
        from PIL import Image
        img = Image.open(path)
        frames = []
        i = 0
        while True:
            try:
                img.seek(i)
                frames.append(np.array(img))
                i += 1
            except EOFError:
                break
        if not frames:
            raise RuntimeError(f"Failed to read TIFF stack: {path}")
        return np.stack(frames, axis=0)


def read_cmap(path: str) -> np.ndarray:
    """
    Read Cmap:
    - .tif/.tiff: read as 3D stack
    - .mat: try common variable names; otherwise choose the largest numeric ndarray
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in [".tif", ".tiff"]:
        return read_tiff_stack(path).astype(np.float32)

    if ext == ".mat":
        try:
            import scipy.io as sio
        except Exception as e:
            raise RuntimeError(
                "You provided a .mat Cmap but scipy is not installed.\n"
                "Install: python -m pip install scipy\n"
            ) from e

        d = sio.loadmat(path)

        # Prefer common variable names
        arr = None
        for key in ["Cmap", "cmap", "CMap", "CMAP"]:
            if key in d and isinstance(d[key], np.ndarray):
                arr = d[key]
                break

        # Otherwise find the largest numeric ndarray
        if arr is None:
            candidates = [(k, v) for k, v in d.items()
                          if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number)]
            if not candidates:
                raise RuntimeError("No numeric ndarray found in the .mat file for Cmap.")
            arr = max(candidates, key=lambda kv: kv[1].size)[1]

        arr = np.array(arr, dtype=np.float32)
        if arr.ndim != 3:
            raise RuntimeError(f"Cmap in .mat is not 3D. shape={arr.shape}")
        return arr

    raise RuntimeError(f"Unsupported Cmap file type: {path}")


# -----------------------------
# Crop around solid bbox (speed-up & remove empty space)
# -----------------------------
def crop_to_solid_bbox(vol: np.ndarray, solid_val: int = 1, pad: int = 10):
    """
    Crop around the bounding box of the solid region, plus padding.
    Returns (cropped_volume, slices)
    """
    solid = (vol == solid_val)
    coords = np.argwhere(solid)
    if coords.size == 0:
        sl = (slice(0, vol.shape[0]), slice(0, vol.shape[1]), slice(0, vol.shape[2]))
        return vol, sl

    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1
    z0 = max(0, z0 - pad); y0 = max(0, y0 - pad); x0 = max(0, x0 - pad)
    z1 = min(vol.shape[0], z1 + pad); y1 = min(vol.shape[1], y1 + pad); x1 = min(vol.shape[2], x1 + pad)

    sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
    return vol[sl], sl


# -----------------------------
# Randomly sample seeds inside pores
# -----------------------------
def sample_pore_seeds(vol: np.ndarray, n: int, rng: np.random.Generator, pore_val: int = 0) -> np.ndarray:
    """
    Sample n random seed points inside pore voxels (value==0).
    Output float coordinates (z,y,x) with small random jitter.
    """
    zdim, ydim, xdim = vol.shape
    out = np.empty((n, 3), dtype=np.float32)
    filled = 0
    while filled < n:
        batch = max(3000, (n - filled) * 8)
        zs = rng.integers(0, zdim, size=batch, dtype=np.int32)
        ys = rng.integers(0, ydim, size=batch, dtype=np.int32)
        xs = rng.integers(0, xdim, size=batch, dtype=np.int32)
        mask = (vol[zs, ys, xs] == pore_val)
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        take = min(idx.size, n - filled)
        sel = idx[:take]
        out[filled:filled+take, 0] = zs[sel]
        out[filled:filled+take, 1] = ys[sel]
        out[filled:filled+take, 2] = xs[sel]
        filled += take

    # jitter seeds so they are not all exactly at voxel centers
    out += rng.random(out.shape, dtype=np.float32) - 0.5
    return out


# -----------------------------
# Vector field from Cmap: v = -grad(Cmap)
# -----------------------------
def downsample(vol: np.ndarray, ds: int) -> np.ndarray:
    """Simple stride-based downsampling."""
    if ds <= 1:
        return vol
    return vol[::ds, ::ds, ::ds]


def compute_vector_field_from_cmap(cmap: np.ndarray, solid_mask: np.ndarray, smooth_sigma: float = 1.0) -> np.ndarray:
    """
    v = -grad(Cmap) in (z,y,x) coordinates => v = (vz, vy, vx)
    solid_mask=True -> set v=0 inside solids
    """
    try:
        from skimage.filters import gaussian
        cmap_s = gaussian(cmap, sigma=smooth_sigma, preserve_range=True).astype(np.float32)
    except Exception:
        cmap_s = cmap.astype(np.float32)

    dz, dy, dx = np.gradient(cmap_s)
    V = np.stack([-dz, -dy, -dx], axis=-1).astype(np.float32)  # (z,y,x,3)
    V[solid_mask] = 0.0
    return V


def trilinear_sample_vec(V: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Trilinear sampling of V(z,y,x,3) at float position p=(z,y,x)."""
    z, y, x = p
    z0 = int(np.floor(z)); y0 = int(np.floor(y)); x0 = int(np.floor(x))
    z1 = z0 + 1; y1 = y0 + 1; x1 = x0 + 1

    Z, Y, X, _ = V.shape
    if z0 < 0 or y0 < 0 or x0 < 0 or z1 >= Z or y1 >= Y or x1 >= X:
        return np.zeros(3, dtype=np.float32)

    dz = z - z0; dy = y - y0; dx = x - x0

    c000 = V[z0, y0, x0]; c001 = V[z0, y0, x1]
    c010 = V[z0, y1, x0]; c011 = V[z0, y1, x1]
    c100 = V[z1, y0, x0]; c101 = V[z1, y0, x1]
    c110 = V[z1, y1, x0]; c111 = V[z1, y1, x1]

    c00 = c000 * (1 - dx) + c001 * dx
    c01 = c010 * (1 - dx) + c011 * dx
    c10 = c100 * (1 - dx) + c101 * dx
    c11 = c110 * (1 - dx) + c111 * dx

    c0 = c00 * (1 - dy) + c01 * dy
    c1 = c10 * (1 - dy) + c11 * dy

    c = c0 * (1 - dz) + c1 * dz
    return c.astype(np.float32)


def is_pore(vol: np.ndarray, p: np.ndarray, pore_val: int = 0) -> bool:
    """Check whether p is inside pore (nearest-voxel test)."""
    zi = int(np.rint(p[0])); yi = int(np.rint(p[1])); xi = int(np.rint(p[2]))
    if zi < 0 or yi < 0 or xi < 0 or zi >= vol.shape[0] or yi >= vol.shape[1] or xi >= vol.shape[2]:
        return False
    return vol[zi, yi, xi] == pore_val


def integrate_streamline(vol: np.ndarray, V: np.ndarray, seed: np.ndarray,
                         step: float, n_steps: int, sign: float,
                         pore_val: int = 0, v_eps: float = 1e-6):
    """
    Integrate one streamline from a seed:
    - sign=+1 forward, sign=-1 backward
    Stop if: speed too small OR hit solid/boundary.
    """
    pts = []
    p = seed.astype(np.float32).copy()
    if not is_pore(vol, p, pore_val=pore_val):
        return pts

    for _ in range(n_steps):
        v = trilinear_sample_vec(V, p)
        speed = float(np.linalg.norm(v))
        if speed < v_eps:
            break
        p_next = p + (sign * step) * (v / speed)
        if not is_pore(vol, p_next, pore_val=pore_val):
            break
        pts.append(p.copy())
        p = p_next

    return pts


def build_streamlines(vol: np.ndarray, V: np.ndarray, seeds: np.ndarray,
                      step: float, n_steps: int, min_points: int = 30, pore_val: int = 0):
    """Build bidirectional streamlines and filter out very short ones."""
    lines = []
    for s in seeds:
        back = integrate_streamline(vol, V, s, step, n_steps, sign=-1.0, pore_val=pore_val)
        fwd  = integrate_streamline(vol, V, s, step, n_steps, sign=+1.0, pore_val=pore_val)
        if not fwd and not back:
            continue
        pts = back[::-1] + [s.astype(np.float32)] + fwd
        if len(pts) < min_points:
            continue
        lines.append(np.stack(pts, axis=0))
    return lines


# -----------------------------
# Solid surface mesh (semi-transparent skeleton)
# Removes boundary shell to avoid drawing ROI box edges.
# -----------------------------
def build_solid_surface(vol: np.ndarray, voxel_um: float, ds_mesh: int = 4, step_size: int = 2,
                        remove_boundary_shell: bool = True) -> np.ndarray:
    from skimage.measure import marching_cubes

    v = vol[::ds_mesh, ::ds_mesh, ::ds_mesh]
    solid = (v == 1).astype(np.uint8)

    verts_zyx, faces, _, _ = marching_cubes(solid, level=0.5, step_size=step_size)

    if remove_boundary_shell:
        Z, Y, X = solid.shape
        eps = 1e-3
        on_boundary = (
            (verts_zyx[:, 0] <= eps) | (verts_zyx[:, 0] >= (Z - 1 - eps)) |
            (verts_zyx[:, 1] <= eps) | (verts_zyx[:, 1] >= (Y - 1 - eps)) |
            (verts_zyx[:, 2] <= eps) | (verts_zyx[:, 2] >= (X - 1 - eps))
        )
        face_has_boundary_vert = on_boundary[faces].any(axis=1)
        faces = faces[~face_has_boundary_vert]

    # Convert from zyx to xyz and scale to micrometers
    scale = voxel_um * ds_mesh
    verts_xyz_um = verts_zyx[:, [2, 1, 0]] * scale
    tris = verts_xyz_um[faces]
    return tris


# -----------------------------
# Convert matplotlib canvas to RGB image (ensure even W/H for yuv420)
# -----------------------------
def canvas_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    if hasattr(fig.canvas, "buffer_rgba"):
        rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    else:
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        rgb = buf[:, :, 1:4].copy()

    h, w = rgb.shape[:2]
    return rgb[:h - (h % 2), :w - (w % 2), :]


def main():
    # Keep CLI options available, but CONFIG is the primary editing point.
    parser = argparse.ArgumentParser(description="3D streamline video renderer (edit CONFIG for defaults).")
    parser.add_argument("--binary", default=CONFIG["BINARY_PATH"])
    parser.add_argument("--cmap", default=CONFIG["CMAP_PATH"])
    parser.add_argument("--output", default=CONFIG["OUTPUT_MP4"])
    parser.add_argument("--voxel_um", type=float, default=CONFIG["VOXEL_UM"])

    parser.add_argument("--duration", type=float, default=CONFIG["DURATION"])
    parser.add_argument("--fps", type=int, default=CONFIG["FPS"])
    parser.add_argument("--width", type=int, default=CONFIG["WIDTH"])
    parser.add_argument("--height", type=int, default=CONFIG["HEIGHT"])
    parser.add_argument("--dpi", type=int, default=CONFIG["DPI"])

    parser.add_argument("--solid_alpha", type=float, default=CONFIG["SOLID_ALPHA"])
    parser.add_argument("--ds_mesh", type=int, default=CONFIG["DS_MESH"])
    parser.add_argument("--mesh_step", type=int, default=CONFIG["MESH_STEP"])

    parser.add_argument("--ds_field", type=int, default=CONFIG["DS_FIELD"])
    parser.add_argument("--smooth_sigma", type=float, default=CONFIG["SMOOTH_SIGMA"])

    parser.add_argument("--n_lines", type=int, default=CONFIG["N_LINES"])
    parser.add_argument("--seed", type=int, default=CONFIG["SEED"])

    parser.add_argument("--step", type=float, default=CONFIG["STEP"])
    parser.add_argument("--n_steps", type=int, default=CONFIG["N_STEPS"])
    parser.add_argument("--min_points", type=int, default=CONFIG["MIN_POINTS"])

    parser.add_argument("--line_alpha", type=float, default=CONFIG["LINE_ALPHA"])
    parser.add_argument("--line_width", type=float, default=CONFIG["LINE_WIDTH"])

    # Glow: allow override both ways
    glow_group = parser.add_mutually_exclusive_group()
    glow_group.add_argument("--enable_glow", dest="enable_glow", action="store_true",
                            help="Enable comet-tail glow (blue moving dots).")
    glow_group.add_argument("--no_glow", dest="enable_glow", action="store_false",
                            help="Disable comet-tail glow.")
    parser.set_defaults(enable_glow=CONFIG["ENABLE_GLOW"])

    parser.add_argument("--glow_window", type=int, default=CONFIG["GLOW_WINDOW"])
    parser.add_argument("--glow_speed", type=float, default=CONFIG["GLOW_SPEED"])
    parser.add_argument("--glow_size", type=float, default=CONFIG["GLOW_SIZE"])

    parser.add_argument("--elev", type=float, default=CONFIG["ELEV"])
    parser.add_argument("--azim0", type=float, default=CONFIG["AZIM0"])
    parser.add_argument("--azim_span", type=float, default=CONFIG["AZIM_SPAN"])
    parser.add_argument("--zoom_amp", type=float, default=CONFIG["ZOOM_AMP"])

    # Crop: allow override both ways
    crop_group = parser.add_mutually_exclusive_group()
    crop_group.add_argument("--no_crop", dest="no_crop", action="store_true", help="Disable auto-crop.")
    crop_group.add_argument("--crop", dest="no_crop", action="store_false", help="Enable auto-crop.")
    parser.set_defaults(no_crop=CONFIG["NO_CROP"])

    args = parser.parse_args()

    args.binary = str(Path(args.binary).resolve())
    args.cmap = str(Path(args.cmap).resolve())
    args.output = str(Path(args.output).resolve())

    if not Path(args.binary).exists():
        raise FileNotFoundError(f"Binary file not found: {args.binary}")
    if not Path(args.cmap).exists():
        raise FileNotFoundError(f"Cmap file not found: {args.cmap}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

    try:
        import imageio
    except Exception:
        raise RuntimeError("Missing imageio. Install:\n  python -m pip install imageio imageio-ffmpeg\n")

    rng = np.random.default_rng(args.seed)

    # Load volumes
    bin_vol = (read_tiff_stack(args.binary).astype(np.uint8) > 0).astype(np.uint8)
    cmap_vol = read_cmap(args.cmap).astype(np.float32)

    # If shapes mismatch, crop to common smallest shape
    if bin_vol.shape != cmap_vol.shape:
        z = min(bin_vol.shape[0], cmap_vol.shape[0])
        y = min(bin_vol.shape[1], cmap_vol.shape[1])
        x = min(bin_vol.shape[2], cmap_vol.shape[2])
        print(f"[WARN] Shape mismatch. Cropping both to {(z, y, x)}")
        bin_vol = bin_vol[:z, :y, :x]
        cmap_vol = cmap_vol[:z, :y, :x]

    # Optional crop around solid bbox
    if not args.no_crop:
        bin_crop, sl = crop_to_solid_bbox(bin_vol, solid_val=1, pad=12)
        cmap_crop = cmap_vol[sl]
    else:
        bin_crop, cmap_crop = bin_vol, cmap_vol

    print(f"[INFO] volume shape={bin_crop.shape}, pore fraction={float((bin_crop==0).mean()):.3f}")

    # Build solid mesh
    tris = build_solid_surface(
        bin_crop, voxel_um=args.voxel_um, ds_mesh=args.ds_mesh, step_size=args.mesh_step,
        remove_boundary_shell=True
    )
    print(f"[INFO] mesh triangles={tris.shape[0]:,}")

    # Build vector field
    ds = max(1, int(args.ds_field))
    bin_ds = downsample(bin_crop, ds)
    cmap_ds = downsample(cmap_crop, ds)
    V = compute_vector_field_from_cmap(cmap_ds, solid_mask=(bin_ds == 1), smooth_sigma=args.smooth_sigma)

    # Streamlines
    seeds = sample_pore_seeds(bin_ds, int(args.n_lines), rng, pore_val=0)
    lines = build_streamlines(bin_ds, V, seeds, step=float(args.step), n_steps=int(args.n_steps),
                              min_points=int(args.min_points), pore_val=0)
    if not lines:
        raise RuntimeError("No streamlines generated. Try: increase N_LINES, reduce MIN_POINTS, increase SMOOTH_SIGMA.")
    print(f"[INFO] streamlines kept={len(lines)}")

    # Convert to world coords (um) + per-point speed
    scale = args.voxel_um * ds
    lines_world = []
    lines_speed = []
    for ln in lines:
        sp = np.array([np.linalg.norm(trilinear_sample_vec(V, p)) for p in ln], dtype=np.float32)
        lines_speed.append(sp)

        xw = ln[:, 2] * scale
        yw = ln[:, 1] * scale
        zw = ln[:, 0] * scale
        lines_world.append(np.stack([xw, yw, zw], axis=1).astype(np.float32))

    all_sp = np.concatenate(lines_speed)
    vmin = float(np.percentile(all_sp, 5))
    vmax = float(np.percentile(all_sp, 95))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    offsets = rng.random(len(lines_world)).astype(np.float32)

    # Figure
    fig = plt.figure(figsize=(args.width / args.dpi, args.height / args.dpi), dpi=args.dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()

    solid = Poly3DCollection(tris, linewidths=0.0, alpha=args.solid_alpha)
    solid.set_facecolor((0.82, 0.82, 0.82, 1.0))
    ax.add_collection3d(solid)

    zdim, ydim, xdim = bin_crop.shape
    x_um, y_um, z_um = xdim * args.voxel_um, ydim * args.voxel_um, zdim * args.voxel_um
    ax.set_xlim(0, x_um); ax.set_ylim(0, y_um); ax.set_zlim(0, z_um)
    try:
        ax.set_box_aspect((x_um, y_um, z_um))
    except Exception:
        pass

    # Speed-colored streamlines
    for lw, sp in zip(lines_world, lines_speed):
        segs = np.stack([lw[:-1], lw[1:]], axis=1)
        seg_sp = 0.5 * (sp[:-1] + sp[1:])
        lc = Line3DCollection(segs, linewidths=args.line_width, alpha=args.line_alpha, cmap="Blues")
        lc.set_array(np.clip(seg_sp, vmin, vmax))
        lc.set_clim(vmin, vmax)
        ax.add_collection3d(lc)

    # Glow scatter
    glow = ax.scatter([], [], [], s=args.glow_size, alpha=1.0)

    # Camera centers
    cx, cy, cz = x_um * 0.5, y_um * 0.5, z_um * 0.5
    rx0, ry0, rz0 = x_um * 0.5, y_um * 0.5, z_um * 0.5

    # Render video
    n_frames = int(args.duration * args.fps)
    print(f"[INFO] rendering {n_frames} frames -> {args.output}")

    if os.path.exists(args.output):
        try:
            os.remove(args.output)
        except Exception:
            pass

    writer = imageio.get_writer(
        args.output,
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )

    try:
        for f in range(n_frames):
            t = f / max(1, n_frames - 1)

            # Camera rotation
            azim = args.azim0 + args.azim_span * t
            ax.view_init(elev=args.elev, azim=azim)

            # Gentle zoom breathing
            zoom = 1.0 - args.zoom_amp * np.sin(2 * np.pi * t)
            ax.set_xlim(cx - rx0 * zoom, cx + rx0 * zoom)
            ax.set_ylim(cy - ry0 * zoom, cy + ry0 * zoom)
            ax.set_zlim(cz - rz0 * zoom, cz + rz0 * zoom)

            # Update glow points
            if args.enable_glow:
                xs_all, ys_all, zs_all = [], [], []
                sizes_all, cols_all = [], []

                w = int(args.glow_window)
                sigma = max(1.0, w / 2.2)

                for i, lw in enumerate(lines_world):
                    n = lw.shape[0]
                    if n < 6:
                        continue

                    phase = (offsets[i] + args.glow_speed * t) % 1.0
                    idx = int(phase * (n - 1))

                    a = max(0, idx - w)
                    b = min(n, idx + w + 1)
                    seg = lw[a:b]

                    k = np.arange(a, b) - idx
                    wgt = np.exp(-0.5 * (k / sigma) ** 2).astype(np.float32)

                    xs_all.append(seg[:, 0]); ys_all.append(seg[:, 1]); zs_all.append(seg[:, 2])
                    sizes_all.append((args.glow_size * (0.6 + 1.2 * wgt)).astype(np.float32))

                    rgba = np.zeros((seg.shape[0], 4), dtype=np.float32)
                    rgba[:, 0] = 0.10
                    rgba[:, 1] = 0.45
                    rgba[:, 2] = 0.95
                    rgba[:, 3] = 0.10 + 0.90 * wgt
                    cols_all.append(rgba)

                if xs_all:
                    xs = np.concatenate(xs_all); ys = np.concatenate(ys_all); zs = np.concatenate(zs_all)
                    sizes = np.concatenate(sizes_all)
                    cols = np.concatenate(cols_all)
                else:
                    xs = np.array([], dtype=np.float32)
                    ys = np.array([], dtype=np.float32)
                    zs = np.array([], dtype=np.float32)
                    sizes = np.array([], dtype=np.float32)
                    cols = np.zeros((0, 4), dtype=np.float32)

                glow._offsets3d = (xs, ys, zs)
                glow.set_sizes(sizes)
                glow.set_color(cols)
            else:
                glow._offsets3d = (np.array([]), np.array([]), np.array([]))
                glow.set_sizes(np.array([]))
                glow.set_color(np.zeros((0, 4), dtype=np.float32))

            frame = canvas_to_rgb(fig)
            writer.append_data(frame)

            if (f + 1) % 20 == 0:
                print(f"[INFO] frame {f+1}/{n_frames}")

    finally:
        writer.close()
        import matplotlib.pyplot as plt
        plt.close(fig)

    print(f"[DONE] saved: {args.output}")


if __name__ == "__main__":
    main()

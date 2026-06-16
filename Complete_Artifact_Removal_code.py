# Generated from: Complete_Artifact_Removal_code.ipynb
# Converted at: 2026-06-16T03:26:04.978Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# ## Below is the code that I find best to remove artifacts of all types from the Gold Atlas Dataset


# It's a 2 stage process


# ## First one is below


import os
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from skimage import morphology, measure, filters

# ---- tunable parameters (edit if the mask clips anatomy or keeps the couch) ----
PAD_CT = -1000.0          # HU assigned outside the body on CT (air)
PAD_MR = 0.0              # value assigned outside the body on MR
CT_AIR_THRESH = -300.0    # HU below this is "not body" (air/couch)
SECOND_COMPONENT_FRAC = 0.15  # keep a 2nd body component (other leg) only if >= this frac of the largest
MIN_BODY_MINOR_MM = 20.0  # a real body cross-section is thick; thin rails (couch) are rejected
MIN_BODY_AREA_FRAC = 0.01 # drop slices whose body area < this fraction of the image
CLOSING_RADIUS = 3        # morphological closing radius (seal the body contour)
OPEN_RADIUS = 4           # morphological opening radius -> severs thin FOV-edge rings / couch bridges
MARKER_HU_OR_Z = 6.0      # MR boundary blob flagged if > this many MADs above local median
APPLY_N4 = False          # True -> N4-correct the MR halo inside the body (needs SimpleITK)
METAL_HU = 2000.0         # CT voxels above this flag a possible metal implant (warn only)

def _clean_components(fg):
    # close gaps, then OPEN to sever thin bridges (FOV-edge ring / couch) before component selection,
    # keep the largest body component (+ a genuinely large second component, e.g. the other leg),
    # then fill internal holes (bowel gas) so they stay inside the body.
    fg = ndi.binary_closing(fg, structure=morphology.disk(CLOSING_RADIUS))   # seal small gaps
    fg = ndi.binary_opening(fg, structure=morphology.disk(OPEN_RADIUS))      # cut thin ring/couch bridges
    lbl = measure.label(fg)
    if lbl.max() == 0:
        return np.zeros_like(fg, dtype=bool)
    props = {p.label: p for p in measure.regionprops(lbl)}
    largest_lab = max(props, key=lambda k: props[k].area)                    # the body
    largest = props[largest_lab].area
    keep = {largest_lab}                                                     # always keep the body
    for lab, p in props.items():                                            # keep a 2nd part (other leg)
        if lab == largest_lab:
            continue
        if p.area >= SECOND_COMPONENT_FRAC * largest and p.axis_minor_length >= MIN_BODY_MINOR_MM:
            keep.add(lab)                                                    # only if large AND thick
    mask = np.isin(lbl, list(keep))
    return ndi.binary_fill_holes(mask)                                       # bowel gas stays inside the body

def _slice_body_mask_ct(sl):
    return _clean_components(sl > CT_AIR_THRESH)                            # CT: body = above air threshold

def _slice_body_mask_mr(sl):
    nz = sl[sl > 0]
    if nz.size < 50:
        return np.zeros_like(sl, dtype=bool)
    thr = filters.threshold_otsu(nz)                                       # MR: data-driven Otsu threshold
    return _clean_components(sl > 0.5 * thr)

def body_mask_3d(vol, modality):
    masker = _slice_body_mask_ct if modality == "ct" else _slice_body_mask_mr
    mask = np.stack([masker(vol[:, :, z]) for z in range(vol.shape[2])], axis=2)
    return ndi.binary_closing(mask, structure=np.ones((1, 1, 3)))          # seal 1-slice z gaps

def n4_correct(mr_vol, mask):
    try:
        import SimpleITK as sitk
    except ImportError:
        print("[warn] SimpleITK not installed -> skipping N4 (halo still removed by mask).")
        return mr_vol
    img = sitk.Cast(sitk.GetImageFromArray(np.ascontiguousarray(mr_vol.transpose(2,0,1))), sitk.sitkFloat32)
    m   = sitk.GetImageFromArray(np.ascontiguousarray(mask.transpose(2,0,1)).astype(np.uint8))
    out = sitk.GetArrayFromImage(sitk.N4BiasFieldCorrection(img, m)).transpose(1,2,0)
    return out.astype(mr_vol.dtype)

def sanity_checks(ct, mr, mask):
    for z in range(mask.shape[2]):
        m = mask[:, :, z]
        if m.sum() == 0:
            continue
        if m[0,:].any() or m[-1,:].any() or m[:,0].any() or m[:,-1].any():
            print(f"[warn] slice {z}: body touches FOV edge.")
        if measure.label(m).max() > 2:
            print(f"[warn] slice {z}: >2 body components (arm/positioning aid?).")
    if np.nanmax(ct) > METAL_HU:
        print(f"[warn] CT max {np.nanmax(ct):.0f} HU > {METAL_HU} -> possible metal/FOV artifact; inspect.")
    boundary = mask ^ ndi.binary_erosion(mask, iterations=2)
    if boundary.any():
        v = mr[boundary]; med = np.median(v); mad = np.median(np.abs(v-med)) + 1e-6
        if (v > med + MARKER_HU_OR_Z*mad).sum() > 5:
            print("[note] bright MR spot on body boundary (skin/fiducial marker); preserved.")

def keep_slice_range(mask):
    area = mask.reshape(-1, mask.shape[2]).sum(0) / (mask.shape[0]*mask.shape[1])
    valid = np.where(area >= MIN_BODY_AREA_FRAC)[0]
    if valid.size == 0:
        raise RuntimeError("no slice has sufficient body area -- check inputs/orientation.")
    return valid.min(), valid.max() + 1

def remove_artifacts(ct_path, mr_path, out_ct_path, out_mr_path, apply_n4=APPLY_N4, trim_slices=True):
    ct_img, mr_img = nib.load(ct_path), nib.load(mr_path)
    ct = ct_img.get_fdata().astype(np.float32)
    mr = mr_img.get_fdata().astype(np.float32)
    assert ct.shape == mr.shape, f"CT {ct.shape} vs MR {mr.shape} must be paired/aligned."
    mask = body_mask_3d(ct, "ct") | body_mask_3d(mr, "mr")                  # union so neither modality is clipped
    if apply_n4:
        mr = n4_correct(mr, mask)
    sanity_checks(ct, mr, mask)
    ct_out = np.where(mask, ct, PAD_CT).astype(np.float32)                  # couch/edge -> -1000 HU
    mr_out = np.where(mask, mr, PAD_MR).astype(np.float32)                  # halo/ghost -> 0
    if trim_slices:
        lo, hi = keep_slice_range(mask)
        ct_out, mr_out = ct_out[:,:,lo:hi], mr_out[:,:,lo:hi]
        print(f"[info] kept slices [{lo}, {hi}) of {ct.shape[2]}")
    nib.save(nib.Nifti1Image(ct_out, ct_img.affine, ct_img.header), out_ct_path)
    nib.save(nib.Nifti1Image(mr_out, mr_img.affine, mr_img.header), out_mr_path)
    print(f"[done] {os.path.basename(os.path.dirname(out_ct_path))}: wrote cleaned CT + MR")
    return mask

print("Cell P0 OK: remove_artifacts and helpers defined (patched: opening + stricter components)")

# ## The second one is this 


"""
refine_all_patients_fixed.py
============================
Same as refine_all_patients.py, but the final 3D closing no longer erases the
first and last z-slice.

THE ONLY CHANGE is in build_refined_mask(): the line
    refined = ndi.binary_closing(refined, structure=np.ones((1, 1, 3)))
eroded the outermost slices, because SciPy's erosion treats out-of-array
voxels as background (border_value=0). We now pad by one slice in z (edge
replication), close, then crop back — so the end slices keep their content.
"""

import os, glob
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import measure, morphology

# ========== CONFIGURATION ==========
INPUT_ROOT   = "scratch_notebooks/cleaned"
OUTPUT_ROOT  = "scratch_notebooks/brand_new_cleaned_slices"
DIAG_OUT_DIR = "scratch_notebooks/fix_diagnostics"

PAD_CT          = -1000.0
PAD_MR          =    0.0
CT_AIR_THRESH   =  -300.0
CLOSING_RADIUS  =     3
OPEN_RADIUS     =     4
DILATION_MM     =    12.0
TINY_FRAC       =   0.003
MIN_MINOR_MM    =  12.0
SECOND_FRAC     =   0.15

DIAG_START      = 160
DIAG_END        = 230
DIAG_COLS       = 7
CT_WIN_C, CT_WIN_W = 40, 400
CT_LO = CT_WIN_C - CT_WIN_W / 2
CT_HI = CT_WIN_C + CT_WIN_W / 2
# ===================================


def clean_ct_slice(ct_sl, vox_xy):
    fg = ct_sl > CT_AIR_THRESH
    if fg.sum() == 0:
        return np.zeros_like(fg, dtype=bool)
    fg = ndi.binary_closing(fg, structure=morphology.disk(CLOSING_RADIUS))
    fg = ndi.binary_opening(fg,  structure=morphology.disk(OPEN_RADIUS))
    lbl = measure.label(fg)
    if lbl.max() == 0:
        return np.zeros_like(fg, dtype=bool)
    img_area = fg.shape[0] * fg.shape[1]
    props    = {p.label: p for p in measure.regionprops(lbl)}
    good = {}
    for lab, p in props.items():
        if p.area < TINY_FRAC * img_area:
            continue
        if p.axis_minor_length * min(vox_xy) < MIN_MINOR_MM:
            continue
        good[lab] = p.area
    if not good:
        largest = max(props, key=lambda k: props[k].area)
        good = {largest: props[largest].area}
    sorted_good  = sorted(good.items(), key=lambda x: x[1], reverse=True)
    largest_area = sorted_good[0][1]
    keep         = {sorted_good[0][0]}
    for lab, area in sorted_good[1:]:
        if area >= SECOND_FRAC * largest_area:
            keep.add(lab)
        else:
            break
    body = np.isin(lbl, list(keep))
    return ndi.binary_fill_holes(body)


def build_refined_mask(ct_vol, mr_vol, vox):
    nx, ny, nz = ct_vol.shape
    vox_xy     = (float(vox[0]), float(vox[1]))
    dil_px     = max(1, int(round(DILATION_MM / min(vox_xy))))
    print(f"    Dilation: {DILATION_MM} mm = {dil_px} px at {vox_xy} mm/px")
    refined = np.zeros((nx, ny, nz), dtype=bool)
    for z in range(nz):
        ct_body    = clean_ct_slice(ct_vol[:, :, z], vox_xy)
        ct_dilated = ndi.binary_dilation(ct_body, structure=morphology.disk(dil_px))
        mr_clipped = (mr_vol[:, :, z] > 0) & ct_dilated
        combined   = ct_body | mr_clipped
        refined[:, :, z] = ndi.binary_fill_holes(combined)

    # --- FIX: border-safe z-closing so slice 0 and slice nz-1 are NOT erased ---
    padded  = np.pad(refined, ((0, 0), (0, 0), (1, 1)), mode="edge")   # replicate end slices
    padded  = ndi.binary_closing(padded, structure=np.ones((1, 1, 3)))
    refined = padded[:, :, 1:-1]                                       # crop back to nz
    return refined


def save_volumes(ct_vol, mr_vol, mask, ct_img, mr_img, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ct_out = np.where(mask, ct_vol, PAD_CT).astype(np.float32)
    mr_out = np.where(mask, mr_vol, PAD_MR).astype(np.float32)
    nib.save(nib.Nifti1Image(ct_out, ct_img.affine, ct_img.header),
             os.path.join(out_dir, "ct_clean.nii.gz"))
    nib.save(nib.Nifti1Image(mr_out, mr_img.affine, mr_img.header),
             os.path.join(out_dir, "mr_clean.nii.gz"))
    print(f"    Saved -> {out_dir}")


def draw_contour(ax, mask_2d, color, lw=0.9):
    for c in measure.find_contours(mask_2d.astype(float), 0.5):
        ax.plot(c[:, 0], c[:, 1], color=color, linewidth=lw, alpha=1.0)


def make_diag_montage(pid, ct_vol, mr_vol, mask_old, mask_new, nz, out_dir):
    slices = [z for z in range(DIAG_START, min(DIAG_END + 1, nz))]
    if not slices:
        return
    n = len(slices)
    n_rows = int(np.ceil(n / DIAG_COLS))
    n_panels = 4
    n_cols_total = DIAG_COLS * n_panels

    nz_vals = mr_vol[mr_vol > 0]
    mr_vmax = float(np.percentile(nz_vals, 99.5)) if nz_vals.size > 0 else 1.0

    fig, axes = plt.subplots(n_rows, n_cols_total,
                             figsize=(DIAG_COLS * n_panels * 1.4, n_rows * 2.6),
                             dpi=110, squeeze=False)
    fig.patch.set_facecolor("black")

    def show(ax, img, cmap, vmin, vmax, title):
        ax.imshow(img.T, cmap=cmap, vmin=vmin, vmax=vmax,
                  origin="upper", interpolation="nearest")
        ax.set_title(title, fontsize=5.5, color="white", pad=1.5)
        ax.axis("off"); ax.set_facecolor("black")

    def show_with_mask(ax, mr, mask, fill_rgb, contour_color, title):
        ax.imshow(mr.T, cmap="gray", vmin=0, vmax=mr_vmax,
                  origin="upper", interpolation="nearest")
        ov = np.zeros((*mr.T.shape, 4), dtype=float)
        ov[..., 0] = fill_rgb[0]; ov[..., 1] = fill_rgb[1]; ov[..., 2] = fill_rgb[2]
        ov[..., 3] = mask.T.astype(float) * 0.28
        ax.imshow(ov, origin="upper", interpolation="nearest")
        draw_contour(ax, mask, color=contour_color, lw=0.9)
        ax.set_title(title, fontsize=5.5, color="white", pad=1.5)
        ax.axis("off"); ax.set_facecolor("black")

    for i, z in enumerate(slices):
        row = i // DIAG_COLS
        base_col = (i % DIAG_COLS) * n_panels
        ct_sl = ct_vol[:, :, z]; mr_sl = mr_vol[:, :, z]
        show(axes[row, base_col], mr_sl, "gray", 0, mr_vmax, f"MR {z}")
        show_with_mask(axes[row, base_col + 1], mr_sl, mask_old[:, :, z], (0, 1, 1), "#00ffff", f"OLD {z}")
        show(axes[row, base_col + 2], ct_sl, "gray", CT_LO, CT_HI, f"CT {z}")
        show_with_mask(axes[row, base_col + 3], mr_sl, mask_new[:, :, z], (0, 1, 0), "#00ff00", f"NEW {z}")

    used = len(slices) % DIAG_COLS
    if used:
        for col in range(used * n_panels, n_cols_total):
            axes[n_rows - 1, col].axis("off"); axes[n_rows - 1, col].set_facecolor("black")

    fig.suptitle(f"{pid}  slices {DIAG_START}-{DIAG_END}   "
                 "[MR raw] | [OLD mask cyan] | [CT] | [NEW mask green]",
                 fontsize=7, color="white", y=1.002)
    plt.subplots_adjust(wspace=0.02, hspace=0.20, left=0, right=1, top=0.997, bottom=0)
    out_png = os.path.join(out_dir, f"fix_check_{pid}.png")
    plt.savefig(out_png, dpi=110, bbox_inches="tight", facecolor="black", pad_inches=0.03)
    plt.close(fig)
    print(f"    Diagnostic montage -> {out_png}")


def process_patient(pid, ct_path, mr_path, out_dir, diag_dir):
    print(f"\n=== {pid} ===")
    ct_img = nib.load(ct_path); mr_img = nib.load(mr_path)
    ct = ct_img.get_fdata().astype(np.float32)
    mr = mr_img.get_fdata().astype(np.float32)
    nz = ct.shape[2]

    vox = np.abs(np.array(ct_img.header.get_zooms()[:3], dtype=float))
    if vox[0] == 0 or vox[1] == 0:
        vox = np.array([1.0, 1.0, 1.0])
    print(f"  Voxel size : {vox[0]:.2f} x {vox[1]:.2f} x {vox[2]:.2f} mm  |  {nz} slices")

    print("  Deriving old mask ...")
    mask_old = (ct > PAD_CT + 50) | (mr > 0)

    print("  Building refined mask (border-safe) ...")
    mask_new = build_refined_mask(ct, mr, vox)

    # confirm the end slices survived
    end_ok = bool(mask_new[:, :, 0].any() and mask_new[:, :, -1].any())
    print(f"  first/last slice non-empty: {end_ok}  "
          f"(slice0={int(mask_new[:,:,0].sum()):,}, slice{nz-1}={int(mask_new[:,:,-1].sum()):,})")

    old_n = int(mask_old.sum()); new_n = int(mask_new.sum()); diff = old_n - new_n
    print(f"  Old body voxels : {old_n:,}")
    print(f"  New body voxels : {new_n:,}")
    print(f"  Removed         : {diff:+,}  ({100*diff/max(old_n,1):+.2f}%)")

    save_volumes(ct, mr, mask_new, ct_img, mr_img, out_dir)
    if nz > DIAG_START:
        make_diag_montage(pid, ct, mr, mask_old, mask_new, nz, diag_dir)


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(DIAG_OUT_DIR, exist_ok=True)
    patient_dirs = sorted(glob.glob(os.path.join(INPUT_ROOT, "*")))
    if not patient_dirs:
        print(f"No patient folders found under {INPUT_ROOT!r}."); return
    for pdir in patient_dirs:
        pid = os.path.basename(pdir)
        ct_path = os.path.join(pdir, "ct_clean.nii.gz")
        mr_path = os.path.join(pdir, "mr_clean.nii.gz")
        if not (os.path.exists(ct_path) and os.path.exists(mr_path)):
            print(f"[{pid}] missing ct_clean or mr_clean - skipped"); continue
        try:
            process_patient(pid, ct_path, mr_path, os.path.join(OUTPUT_ROOT, pid), DIAG_OUT_DIR)
        except Exception as e:
            print(f"  !! FAILED on {pid}: {e}")
    print(f"\nDONE. Refined volumes in {OUTPUT_ROOT}, diagnostics in {DIAG_OUT_DIR}")


if __name__ == "__main__":
    main()

# ## So just to show a demo on one patient, 1_03_P, before and after each step of artifact removal


# ## A visual montage of every 5 slices of 1_03_P raw from scratch_notebooks/isotropic1mm


import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
iso_root = Path("scratch_notebooks/isotropic1mm")
patient = "1_03_P"
output_dir = Path("lol")
output_dir.mkdir(exist_ok=True)

# Load volumes
ct_path = iso_root / patient / "ct_iso.nii.gz"
mr_path = iso_root / patient / "mri_iso.nii.gz"
if not mr_path.exists():
    mr_path = iso_root / patient / "mr_iso.nii.gz"

ct_data = nib.load(str(ct_path)).get_fdata()
mr_data = nib.load(str(mr_path)).get_fdata()

# Every 5th slice
n_slices = ct_data.shape[2]
slice_indices = list(range(0, n_slices, 5))
print(f"Total slices: {n_slices}, showing {len(slice_indices)} slices (every 5th)")

# Prepare figure: one row per slice, each row has 2 columns (CT, MRI)
rows = len(slice_indices)
fig, axes = plt.subplots(rows, 2, figsize=(10, rows * 4))  # each row 4 inches tall
if rows == 1:
    axes = axes.reshape(1, 2)

# CT window (soft tissue)
vmin_ct, vmax_ct = -150, 250

# MRI scaling (99th percentile)
pos = mr_data[mr_data > 0]
p99 = np.percentile(pos, 99) if pos.size else 1.0

for i, idx in enumerate(slice_indices):
    ax_ct = axes[i, 0]
    ax_mr = axes[i, 1]
    
    ax_ct.imshow(ct_data[:, :, idx], cmap='gray', vmin=vmin_ct, vmax=vmax_ct)
    ax_ct.set_ylabel(f"CT slice {idx}", fontsize=10)
    ax_ct.axis('off')
    
    ax_mr.imshow(mr_data[:, :, idx], cmap='gray', vmin=0, vmax=p99)
    ax_mr.set_ylabel(f"MRI slice {idx}", fontsize=10)
    ax_mr.axis('off')

plt.suptitle(f"{patient}: every 5th slice – one slice per row (CT left, MRI right)", fontsize=14)
plt.tight_layout()
plt.savefig(output_dir / f"{patient}_every5th_vertical.png", dpi=100, bbox_inches='tight')
plt.show()
print(f"Saved tall montage: {output_dir / f'{patient}_every5th_vertical.png'}")

# ## Now applying the first stage of artifact removal and then displaying every 5 slices again for that


import os
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from skimage import morphology, measure, filters
import matplotlib.pyplot as plt
import math
from pathlib import Path

# ---------- 1. Define the cleaning functions (same as your code) ----------
PAD_CT = -1000.0
PAD_MR = 0.0
CT_AIR_THRESH = -300.0
SECOND_COMPONENT_FRAC = 0.15
MIN_BODY_MINOR_MM = 20.0
MIN_BODY_AREA_FRAC = 0.01
CLOSING_RADIUS = 3
OPEN_RADIUS = 4
MARKER_HU_OR_Z = 6.0
APPLY_N4 = False
METAL_HU = 2000.0

def _clean_components(fg):
    fg = ndi.binary_closing(fg, structure=morphology.disk(CLOSING_RADIUS))
    fg = ndi.binary_opening(fg, structure=morphology.disk(OPEN_RADIUS))
    lbl = measure.label(fg)
    if lbl.max() == 0:
        return np.zeros_like(fg, dtype=bool)
    props = {p.label: p for p in measure.regionprops(lbl)}
    largest_lab = max(props, key=lambda k: props[k].area)
    largest = props[largest_lab].area
    keep = {largest_lab}
    for lab, p in props.items():
        if lab == largest_lab:
            continue
        if p.area >= SECOND_COMPONENT_FRAC * largest and p.axis_minor_length >= MIN_BODY_MINOR_MM:
            keep.add(lab)
    mask = np.isin(lbl, list(keep))
    return ndi.binary_fill_holes(mask)

def _slice_body_mask_ct(sl):
    return _clean_components(sl > CT_AIR_THRESH)

def _slice_body_mask_mr(sl):
    nz = sl[sl > 0]
    if nz.size < 50:
        return np.zeros_like(sl, dtype=bool)
    thr = filters.threshold_otsu(nz)
    return _clean_components(sl > 0.5 * thr)

def body_mask_3d(vol, modality):
    masker = _slice_body_mask_ct if modality == "ct" else _slice_body_mask_mr
    mask = np.stack([masker(vol[:, :, z]) for z in range(vol.shape[2])], axis=2)
    return ndi.binary_closing(mask, structure=np.ones((1, 1, 3)))

def keep_slice_range(mask):
    img_area = mask.shape[0] * mask.shape[1]
    areas = mask.reshape(-1, mask.shape[2]).sum(0) / img_area
    valid = np.where(areas >= MIN_BODY_AREA_FRAC)[0]
    if valid.size == 0:
        raise RuntimeError("no slice has sufficient body area")
    return valid.min(), valid.max() + 1

def remove_artifacts(ct_path, mr_path, out_ct_path, out_mr_path, apply_n4=False, trim_slices=True):
    ct_img = nib.load(ct_path)
    mr_img = nib.load(mr_path)
    ct = ct_img.get_fdata().astype(np.float32)
    mr = mr_img.get_fdata().astype(np.float32)
    assert ct.shape == mr.shape, f"Shape mismatch: {ct.shape} vs {mr.shape}"
    mask = body_mask_3d(ct, "ct") | body_mask_3d(mr, "mr")
    if apply_n4:
        # N4 correction not used here; kept for completeness
        pass
    ct_out = np.where(mask, ct, PAD_CT).astype(np.float32)
    mr_out = np.where(mask, mr, PAD_MR).astype(np.float32)
    if trim_slices:
        lo, hi = keep_slice_range(mask)
        ct_out = ct_out[:, :, lo:hi]
        mr_out = mr_out[:, :, lo:hi]
        print(f"[info] kept slices [{lo}, {hi}) of {ct.shape[2]} total")
    nib.save(nib.Nifti1Image(ct_out, ct_img.affine, ct_img.header), out_ct_path)
    nib.save(nib.Nifti1Image(mr_out, mr_img.affine, mr_img.header), out_mr_path)
    print(f"[done] saved {out_ct_path} and {out_mr_path}")
    return mask

# ---------- 2. Apply to patient 1_03_P ----------
patient = "1_03_P"
iso_root = Path("scratch_notebooks/isotropic1mm")
ct_input = iso_root / patient / "ct_iso.nii.gz"
mr_input = iso_root / patient / "mri_iso.nii.gz"
if not mr_input.exists():
    mr_input = iso_root / patient / "mr_iso.nii.gz"

# Output folder
out_folder = Path("1_03_Pphaseone")
out_folder.mkdir(exist_ok=True)
ct_cleaned_path = out_folder / "ct_clean.nii.gz"
mr_cleaned_path = out_folder / "mr_clean.nii.gz"

print(f"Processing {patient}...")
remove_artifacts(str(ct_input), str(mr_input), str(ct_cleaned_path), str(mr_cleaned_path),
                 apply_n4=False, trim_slices=True)

# ---------- 3. Load cleaned volumes and visualise every 5th slice ----------
ct_clean = nib.load(str(ct_cleaned_path)).get_fdata()
mr_clean = nib.load(str(mr_cleaned_path)).get_fdata()

n_slices = ct_clean.shape[2]
slice_indices = list(range(0, n_slices, 5))
print(f"Cleaned volume has {n_slices} slices; showing {len(slice_indices)} every 5th")

# One slice per row, each row has CT (left) and MRI (right)
rows = len(slice_indices)
fig, axes = plt.subplots(rows, 2, figsize=(8, rows * 3.5))   # each row 3.5 inches tall
if rows == 1:
    axes = axes.reshape(1, 2)

# CT window (soft tissue)
vmin_ct, vmax_ct = -150, 250

# MRI is already normalised to [0,1] after cleaning (background 0)
for i, idx in enumerate(slice_indices):
    ax_ct = axes[i, 0]
    ax_mr = axes[i, 1]
    ax_ct.imshow(ct_clean[:, :, idx], cmap='gray', vmin=vmin_ct, vmax=vmax_ct)
    ax_ct.set_ylabel(f"CT {idx}", fontsize=9)
    ax_ct.axis('off')
    ax_mr.imshow(mr_clean[:, :, idx], cmap='gray', vmin=0, vmax=1)
    ax_mr.set_ylabel(f"MRI {idx}", fontsize=9)
    ax_mr.axis('off')

plt.suptitle(f"{patient}: cleaned CT (left) and MRI (right) – every 5th slice", fontsize=12)
plt.tight_layout()
out_fig = out_folder / f"{patient}_cleaned_every5th.png"
plt.savefig(out_fig, dpi=120, bbox_inches='tight')
plt.show()
print(f"Saved visualisation to {out_fig}")

%matplotlib inline
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

# Folder containing cleaned volumes
src_dir = Path("1_03_Pphaseone")
ct_path = src_dir / "ct_clean.nii.gz"
mr_path = src_dir / "mr_clean.nii.gz"

# Load data
ct = nib.load(str(ct_path)).get_fdata().astype(np.float32)
mr = nib.load(str(mr_path)).get_fdata().astype(np.float32)

# Parameters
step = 5
ct_win = (-200, 400)           # soft‑tissue window (HU)

# Scaling functions (same as your working script)
def ct_disp(vol):
    lo, hi = ct_win
    return np.clip((vol - lo) / (hi - lo), 0, 1)

def mr_disp(vol):
    pos = vol[vol > 0]
    p99 = np.percentile(pos, 99) if pos.size else 1.0
    return np.clip(vol / (p99 + 1e-9), 0, 1)

# Apply scaling
ct_scaled = ct_disp(ct)
mr_scaled = mr_disp(mr)

# Indices for every 5th slice
n_slices = ct.shape[2]
slice_indices = list(range(0, n_slices, step))
rows = len(slice_indices)

# Create figure: one row per slice, two columns (CT, MRI)
fig, axes = plt.subplots(rows, 2, figsize=(8, rows * 3.5))
if rows == 1:
    axes = axes.reshape(1, 2)

for i, idx in enumerate(slice_indices):
    # CT (left) – no transpose, default origin='upper'
    axes[i, 0].imshow(ct_scaled[:, :, idx], cmap='gray', vmin=0, vmax=1)
    axes[i, 0].set_ylabel(f"CT {idx}", fontsize=9)
    axes[i, 0].axis('off')
    # MRI (right)
    axes[i, 1].imshow(mr_scaled[:, :, idx], cmap='gray', vmin=0, vmax=1)
    axes[i, 1].set_ylabel(f"MRI {idx}", fontsize=9)
    axes[i, 1].axis('off')

plt.suptitle(f"1_03_P – every {step}th slice (cleaned CT left, MRI right)", fontsize=12)
plt.tight_layout()
plt.show()

# ## Now applying the second/last stage of artifact removal and then displaying every 5 slices again for that


import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import measure, morphology
from pathlib import Path

# ========== CONFIGURATION (patient 1_03_P) ==========
PATIENT = "1_03_P"
INPUT_DIR  = Path("1_03_Pphaseone")        # where ct_clean.nii.gz and mr_clean.nii.gz are
OUTPUT_DIR = Path("1_03_Pphasetwo")        # where refined volumes will be saved

# Parameters (same as in refinement script)
PAD_CT          = -1000.0
PAD_MR          =    0.0
CT_AIR_THRESH   =  -300.0
CLOSING_RADIUS  =     3
OPEN_RADIUS     =     4
DILATION_MM     =    12.0
TINY_FRAC       =   0.003
MIN_MINOR_MM    =  12.0
SECOND_FRAC     =   0.15
# ====================================================

def clean_ct_slice(ct_sl, vox_xy):
    fg = ct_sl > CT_AIR_THRESH
    if fg.sum() == 0:
        return np.zeros_like(fg, dtype=bool)
    fg = ndi.binary_closing(fg, structure=morphology.disk(CLOSING_RADIUS))
    fg = ndi.binary_opening(fg,  structure=morphology.disk(OPEN_RADIUS))
    lbl = measure.label(fg)
    if lbl.max() == 0:
        return np.zeros_like(fg, dtype=bool)
    img_area = fg.shape[0] * fg.shape[1]
    props    = {p.label: p for p in measure.regionprops(lbl)}
    good = {}
    for lab, p in props.items():
        if p.area < TINY_FRAC * img_area:
            continue
        if p.axis_minor_length * min(vox_xy) < MIN_MINOR_MM:
            continue
        good[lab] = p.area
    if not good:
        largest = max(props, key=lambda k: props[k].area)
        good = {largest: props[largest].area}
    sorted_good  = sorted(good.items(), key=lambda x: x[1], reverse=True)
    largest_area = sorted_good[0][1]
    keep         = {sorted_good[0][0]}
    for lab, area in sorted_good[1:]:
        if area >= SECOND_FRAC * largest_area:
            keep.add(lab)
        else:
            break
    body = np.isin(lbl, list(keep))
    return ndi.binary_fill_holes(body)

def build_refined_mask(ct_vol, mr_vol, vox):
    nx, ny, nz = ct_vol.shape
    vox_xy     = (float(vox[0]), float(vox[1]))
    dil_px     = max(1, int(round(DILATION_MM / min(vox_xy))))
    refined = np.zeros((nx, ny, nz), dtype=bool)
    for z in range(nz):
        ct_body    = clean_ct_slice(ct_vol[:, :, z], vox_xy)
        ct_dilated = ndi.binary_dilation(ct_body, structure=morphology.disk(dil_px))
        mr_clipped = (mr_vol[:, :, z] > 0) & ct_dilated
        combined   = ct_body | mr_clipped
        refined[:, :, z] = ndi.binary_fill_holes(combined)
    # Border‑safe z‑closing (preserves end slices)
    padded = np.pad(refined, ((0,0),(0,0),(1,1)), mode="edge")
    padded = ndi.binary_closing(padded, structure=np.ones((1,1,3)))
    return padded[:, :, 1:-1]

def refine_patient(input_dir, output_dir):
    ct_path = input_dir / "ct_clean.nii.gz"
    mr_path = input_dir / "mr_clean.nii.gz"
    if not (ct_path.exists() and mr_path.exists()):
        raise FileNotFoundError(f"Missing files in {input_dir}")
    ct_img = nib.load(str(ct_path))
    mr_img = nib.load(str(mr_path))
    ct = ct_img.get_fdata().astype(np.float32)
    mr = mr_img.get_fdata().astype(np.float32)
    vox = np.abs(np.array(ct_img.header.get_zooms()[:3], dtype=float))
    if vox[0]==0 or vox[1]==0:
        vox = np.array([1.0,1.0,1.0])
    mask_new = build_refined_mask(ct, mr, vox)
    output_dir.mkdir(parents=True, exist_ok=True)
    ct_out = np.where(mask_new, ct, PAD_CT).astype(np.float32)
    mr_out = np.where(mask_new, mr, PAD_MR).astype(np.float32)
    nib.save(nib.Nifti1Image(ct_out, ct_img.affine, ct_img.header), output_dir / "ct_clean.nii.gz")
    nib.save(nib.Nifti1Image(mr_out, mr_img.affine, mr_img.header), output_dir / "mr_clean.nii.gz")
    print(f"Refined -> {output_dir}")
    return ct_out, mr_out

# ---- 1. Run refinement from 1_03_Pphaseone to 1_03_Pphasetwo ----
print(f"Reading cleaned volumes from {INPUT_DIR}...")
ref_ct, ref_mr = refine_patient(INPUT_DIR, OUTPUT_DIR)

# ---- 2. Visualise every 5th slice (vertical stack) ----
step = 5
ct_win = (-200, 400)          # soft‑tissue window (HU)

def ct_disp(vol):
    lo, hi = ct_win
    return np.clip((vol - lo) / (hi - lo), 0, 1)

def mr_disp(vol):
    pos = vol[vol > 0]
    p99 = np.percentile(pos, 99) if pos.size else 1.0
    return np.clip(vol / (p99 + 1e-9), 0, 1)

ct_scaled = ct_disp(ref_ct)
mr_scaled = mr_disp(ref_mr)

n_slices = ref_ct.shape[2]
indices = list(range(0, n_slices, step))
rows = len(indices)

fig, axes = plt.subplots(rows, 2, figsize=(8, rows * 3.5))
if rows == 1:
    axes = axes.reshape(1, 2)

for i, idx in enumerate(indices):
    axes[i, 0].imshow(ct_scaled[:, :, idx], cmap='gray', vmin=0, vmax=1)
    axes[i, 0].set_ylabel(f"CT {idx}", fontsize=9)
    axes[i, 0].axis('off')
    axes[i, 1].imshow(mr_scaled[:, :, idx], cmap='gray', vmin=0, vmax=1)
    axes[i, 1].set_ylabel(f"MRI {idx}", fontsize=9)
    axes[i, 1].axis('off')

plt.suptitle(f"{PATIENT} refined (phase two) – every {step}th slice (CT left, MRI right)", fontsize=12)
plt.tight_layout()
plt.show()

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Path to refined volumes
phase_two = Path("1_03_Pphasetwo")
ct_ref = nib.load(phase_two / "ct_clean.nii.gz").get_fdata().astype(np.float32)
mr_ref = nib.load(phase_two / "mr_clean.nii.gz").get_fdata().astype(np.float32)

# Derive binary mask (body = 1, background = 0) from MRI or CT
mask = (mr_ref > 0).astype(np.float32)   # because cleaned MRI background is 0
# Alternatively from CT: mask = (ct_ref > -950).astype(np.float32)

# Parameters
step = 5
ct_win = (-200, 400)          # soft‑tissue window (HU)

def ct_disp(vol):
    lo, hi = ct_win
    return np.clip((vol - lo) / (hi - lo), 0, 1)

def mr_disp(vol):
    pos = vol[vol > 0]
    p99 = np.percentile(pos, 99) if pos.size else 1.0
    return np.clip(vol / (p99 + 1e-9), 0, 1)

ct_scaled = ct_disp(ct_ref)
mr_scaled = mr_disp(mr_ref)

n_slices = ct_ref.shape[2]
indices = list(range(0, n_slices, step))
rows = len(indices)

# Create figure: each row has 3 columns (CT, MRI, mask)
fig, axes = plt.subplots(rows, 3, figsize=(12, rows * 3.5))
if rows == 1:
    axes = axes.reshape(1, 3)

for i, idx in enumerate(indices):
    # CT (grayscale)
    axes[i, 0].imshow(ct_scaled[:, :, idx], cmap='gray', vmin=0, vmax=1)
    axes[i, 0].set_ylabel(f"CT {idx}", fontsize=9)
    axes[i, 0].axis('off')
    # MRI (grayscale)
    axes[i, 1].imshow(mr_scaled[:, :, idx], cmap='gray', vmin=0, vmax=1)
    axes[i, 1].set_ylabel(f"MRI {idx}", fontsize=9)
    axes[i, 1].axis('off')
    # Binary mask
    axes[i, 2].imshow(mask[:, :, idx], cmap='gray', vmin=0, vmax=1)
    axes[i, 2].set_ylabel(f"Mask {idx}", fontsize=9)
    axes[i, 2].axis('off')

plt.suptitle(f"1_03_Pphasetwo – every {step}th slice (CT | MRI | binary mask)", fontsize=12)
plt.tight_layout()
plt.show()
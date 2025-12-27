# Suppose you kept the raw sample dicts in `all_data` (the list you built before creating the Dataset)
# and you have `benefit_generator` + `lama` already wired earlier.
# ============================
# Quick evaluation (PSNR/SSIM)
# ============================
import os, math, numpy as np, torch, torch.nn.functional as F, cv2
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import argparse
from tqdm import tqdm
from pipeline_difix import DifixPipeline
from diffusers.utils import load_image
import torch


def load_npy_chw(path: str) -> torch.Tensor:
    """
    Load a saved .npy array and return a float32 torch tensor in [C,H,W].
    Accepts [C,H,W], [H,W,C], or [H,W] arrays.
    """
    try:
        arr = np.load(path)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)

        if arr.ndim == 2:
            t = torch.from_numpy(arr).unsqueeze(0)             # [1,H,W]
        elif arr.ndim == 3:
            if arr.shape[0] in (1, 3):                         # already CHW
                t = torch.from_numpy(arr)
            elif arr.shape[-1] in (1, 3):                      # HWC -> CHW
                t = torch.from_numpy(arr).permute(2, 0, 1)
            else:
                raise ValueError(f"Unexpected 3D shape for {path}: {arr.shape}")
        else:
            raise ValueError(f"Unexpected ndim for {path}: {arr.ndim}")
        return t
    except Exception as e:
        print(f"Error loading {path}: {e}")
        raise

class ErrorMapGenerator:
    """Generate ground truth error maps by comparing rendered with GT images"""
    
    @staticmethod
    def generate_error_map(rendered_img, gt_img, method='combined'):
        """
        Generate error map by comparing rendered image with ground truth
        
        Methods:
        - 'l1': L1 distance
        - 'l2': L2 distance  
        - 'perceptual': Perceptual error using gradients
        - 'ssim': Structural dissimilarity (1 - SSIM)
        - 'combined': Weighted combination of multiple error metrics
        
        Returns:
            Error map in [0, 1] range
        """
        rendered = rendered_img.detach().cpu().numpy() if torch.is_tensor(rendered_img) else rendered_img
        gt = gt_img.detach().cpu().numpy() if torch.is_tensor(gt_img) else gt_img
        
        # Ensure proper shape (H, W, C)
        if rendered.shape[0] == 3:
            rendered = rendered.transpose(1, 2, 0)
        if gt.shape[0] == 3:
            gt = gt.transpose(1, 2, 0)
        
        if method == 'l1':
            error = np.abs(rendered - gt).mean(axis=2)
            error = np.clip(error, 0, 1)
            
        elif method == 'l2':
            error = np.sqrt(((rendered - gt) ** 2).mean(axis=2))
            error = np.clip(error, 0, 1)
            
        elif method == 'perceptual':
            rendered_gray = cv2.cvtColor((rendered * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
            gt_gray = cv2.cvtColor((gt * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
            
            grad_x_r = cv2.Sobel(rendered_gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y_r = cv2.Sobel(rendered_gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_x_gt = cv2.Sobel(gt_gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y_gt = cv2.Sobel(gt_gray, cv2.CV_64F, 0, 1, ksize=3)
            
            grad_diff = np.sqrt((grad_x_r - grad_x_gt)**2 + (grad_y_r - grad_y_gt)**2)
            error = np.clip(grad_diff, 0, 1)
            
        elif method == 'ssim':
            ssim_map = structural_similarity(rendered, gt, channel_axis=2, 
                                           data_range=1.0, win_size=11, full=True)[1]
            error = 1.0 - ssim_map.mean(axis=2)
            
        elif method == 'combined':
            # IMPROVED: More balanced weights
            
            # 1. Color error (L1 in RGB)
            color_error = np.abs(rendered - gt).mean(axis=2)
            
            # 2. Structural error (using local SSIM)
            ssim_map = structural_similarity(rendered, gt, channel_axis=2,
                                           data_range=1.0, win_size=11, full=True)[1]
            structural_error = 1.0 - ssim_map.mean(axis=2)
            
            # 3. Edge error
            rendered_gray = cv2.cvtColor((rendered * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
            gt_gray = cv2.cvtColor((gt * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
            
            edges_r = cv2.Canny((rendered_gray * 255).astype(np.uint8), 50, 150) / 255.0
            edges_gt = cv2.Canny((gt_gray * 255).astype(np.uint8), 50, 150) / 255.0
            edge_error = np.abs(edges_r - edges_gt)
            
            # 4. Perceptual error (using Laplacian)
            laplacian_r = cv2.Laplacian(rendered_gray, cv2.CV_64F)
            laplacian_gt = cv2.Laplacian(gt_gray, cv2.CV_64F)
            perceptual_error = np.abs(laplacian_r - laplacian_gt)
            perceptual_error = np.clip(perceptual_error, 0, 1)
            

            error = (0.85 * color_error + 
                    0.15 * structural_error + 
                    0.00 * edge_error +
                    0.00 * perceptual_error)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Apply smoothing to reduce noise
        error = cv2.GaussianBlur(error.astype(np.float32), (5, 5), 1.0)
        error = np.clip(error, 0, 1)
        
        return torch.from_numpy(error).float()

def as_numpy(img: torch.Tensor | np.ndarray) -> np.ndarray:
    """Ensure HxWxC numpy float32 in [0,1] for images; HxW for single-channel."""
    if isinstance(img, torch.Tensor):
        t = img.detach().cpu()
        if t.ndim == 3 and t.shape[0] in (1, 3):  # CHW -> HWC
            t = t.permute(1, 2, 0)
        img_np = t.numpy()
    else:
        img_np = img
    # Normalize uint8 to [0,1]
    if img_np.dtype == np.uint8:
        img_np = img_np.astype(np.float32) / 255.0
    return img_np.astype(np.float32)


def chw(t: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Ensure CHW torch float32 in [0,1]."""
    if isinstance(t, np.ndarray):
        arr = t
        if arr.ndim == 2:
            arr = arr[None, ...]  # H W -> 1 H W
        elif arr.ndim == 3 and arr.shape[2] in (1, 3):  # H W C -> C H W
            arr = arr.transpose(2, 0, 1)
        ten = torch.from_numpy(arr)
    else:
        ten = t
    ten = ten.float()
    if ten.dtype != torch.float32:
        ten = ten.float()
    return ten

class InpaintingEffectivenessAnalyzer:
    @staticmethod
    def compute_error_maps(original_rendered: torch.Tensor | np.ndarray,
                           inpainted_image: torch.Tensor | np.ndarray,
                           gt_image: torch.Tensor | np.ndarray,
                           method: str = "combined") -> Tuple[np.ndarray, np.ndarray]:
        orig_np = as_numpy(original_rendered)
        inp_np = as_numpy(inpainted_image)
        gt_np = as_numpy(gt_image)
        e0 = ErrorMapGenerator.generate_error_map(orig_np, gt_np, method=method)
        e1 = ErrorMapGenerator.generate_error_map(inp_np, gt_np, method=method)
        return e0, e1

    @staticmethod
    def compute_metrics(original: torch.Tensor | np.ndarray,
                         inpainted: torch.Tensor | np.ndarray,
                         gt: torch.Tensor | np.ndarray,
                        #  mask: torch.Tensor | np.ndarray,
                         ) -> Dict:
        o = as_numpy(original)
        i = as_numpy(inpainted)
        g = as_numpy(gt)
        # m = as_numpy(mask)
        # if m.ndim == 3:
        #     m = np.squeeze(m)
    
        # mb = m > 0.5
        # nmb = ~mb

        def mse(a, b):
            return float(np.mean((a - b) ** 2))

        def mae(a, b):
            return float(np.mean(np.abs(a - b)))

        metrics = {
            "original": {
                "mse": mse(o, g),
                "mae": mae(o, g),
                "psnr": float(psnr(g, o, data_range=1.0)),
                "ssim": float(ssim(g, o, channel_axis=2, data_range=1.0)),
            },
            "inpainted": {
                "mse": mse(i, g),
                "mae": mae(i, g),
                "psnr": float(psnr(g, i, data_range=1.0)),
                "ssim": float(ssim(g, i, channel_axis=2, data_range=1.0)),
            },
        }

        # if mb.sum() > 0:
        #     metrics["original_masked"] = {"mse": mse(o[mb], g[mb]), "mae": mae(o[mb], g[mb])}
        #     metrics["inpainted_masked"] = {"mse": mse(i[mb], g[mb]), "mae": mae(i[mb], g[mb])}

        # if nmb.sum() > 0:
        #     metrics["original_unmasked"] = {"mse": mse(o[nmb], g[nmb]), "mae": mae(o[nmb], g[nmb])}
        #     metrics["inpainted_unmasked"] = {"mse": mse(i[nmb], g[nmb]), "mae": mae(i[nmb], g[nmb])}

        metrics["improvement"] = {
            "mse_reduction": metrics["original"]["mse"] - metrics["inpainted"]["mse"],
            "mae_reduction": metrics["original"]["mae"] - metrics["inpainted"]["mae"],
            "psnr_gain": metrics["inpainted"]["psnr"] - metrics["original"]["psnr"],
            "ssim_gain": metrics["inpainted"]["ssim"] - metrics["original"]["ssim"],
        }

        if "original_masked" in metrics:
            metrics["improvement_masked"] = {
                "mse_reduction": metrics["original_masked"]["mse"] - metrics["inpainted_masked"]["mse"],
                "mae_reduction": metrics["original_masked"]["mae"] - metrics["inpainted_masked"]["mae"],
            }
        return metrics

    @staticmethod
    def improvement_map(original_error: torch.Tensor | np.ndarray,
                        inpainted_error: torch.Tensor | np.ndarray
                        # mask: torch.Tensor | np.ndarray
                        ) -> np.ndarray:
        e0 = as_numpy(original_error)
        e1 = as_numpy(inpainted_error)
        # m = as_numpy(mask)
        if e0.ndim == 3: e0 = np.squeeze(e0)
        if e1.ndim == 3: e1 = np.squeeze(e1)
        # if m.ndim == 3: m = np.squeeze(m)
        imp = e0 - e1
        # imp[m <= 0.5] = 0
        return imp


def _create_metrics_table(ax, metrics: Dict) -> None:
    ax.axis("off")
    headers = ["METRIC", "ORIGINAL", "INPAINTED", "IMPROVEMENT", "STATUS"]

    def row(metric: str, a: float, b: float, imp: float, up_is_good: bool = True):
        ok = (imp > 0) if up_is_good else (imp < 0)
        return [metric, f"{a:.4f}", f"{b:.4f}", f"{imp:+.4f}", "✓" if ok else "✗"]

    rows = [
        row("MSE", metrics["original"]["mse"], metrics["inpainted"]["mse"],
            metrics["improvement"]["mse_reduction"], up_is_good=True),
        row("MAE", metrics["original"]["mae"], metrics["inpainted"]["mae"],
            metrics["improvement"]["mae_reduction"], up_is_good=True),
        row("PSNR", metrics["original"]["psnr"], metrics["inpainted"]["psnr"],
            metrics["improvement"]["psnr_gain"], up_is_good=True),
        row("SSIM", metrics["original"]["ssim"], metrics["inpainted"]["ssim"],
            metrics["improvement"]["ssim_gain"], up_is_good=True),
    ]

    if "improvement_masked" in metrics:
        rows += [
            ["────────", "────────", "────────", "──────────", "────"],
            ["MASKED REGIONS ONLY", "", "", "", ""],
            row("MSE (Mask)", metrics["original_masked"]["mse"], metrics["inpainted_masked"]["mse"],
                metrics["improvement_masked"]["mse_reduction"], up_is_good=True),
            row("MAE (Mask)", metrics["original_masked"]["mae"], metrics["inpainted_masked"]["mae"],
                metrics["improvement_masked"]["mae_reduction"], up_is_good=True),
        ]

    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center",
                     colWidths=[0.18, 0.18, 0.18, 0.18, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code improvement cells
    for i in range(len(rows)):
        if len(rows[i]) > 4:
            status = rows[i][4]
            color = '#90EE90' if status == '✓' else '#FFB6C1'  # Light green/red
            table[(i+1, 4)].set_facecolor(color)

def _create_improvement_hist(ax, improvement_map: np.ndarray, mask: np.ndarray) -> None:
    """Create histogram of error improvements in masked regions"""
    
    # Get improvements only in masked regions
    mask_binary = (mask > 0.5)
    masked_improvements = improvement_map[mask_binary]
    
    if len(masked_improvements) == 0:
        ax.text(0.5, 0.5, 'No masked regions', transform=ax.transAxes, 
               ha='center', va='center', fontsize=12)
        ax.set_title('Error Improvement Distribution')
        return
    
    # Create histogram
    n_bins = 30
    counts, bins, patches = ax.hist(masked_improvements, bins=n_bins, alpha=0.7, edgecolor='black')
    
    # Color bars: green for positive (improvement), red for negative (degradation)
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center > 0:
            patch.set_facecolor('#4CAF50')  # Green for improvements
        else:
            patch.set_facecolor('#F44336')  # Red for degradations
    
    # Add vertical line at zero
    ax.axvline(0, color='black', linestyle='--', linewidth=2, label='No Change')
    
    # Add statistics text
    positive_pixels = np.sum(masked_improvements > 0)
    negative_pixels = np.sum(masked_improvements < 0)
    total_pixels = len(masked_improvements)
    
    improvement_ratio = positive_pixels / total_pixels * 100
    mean_improvement = np.mean(masked_improvements)
    
    stats_text = f'Improved: {improvement_ratio:.1f}%\n'
    stats_text += f'Mean Δ: {mean_improvement:.4f}\n'
    stats_text += f'Pixels: {total_pixels}'
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Error Change (Positive = Better)')
    ax.set_ylabel('Pixel Count')
    ax.set_title('Error Improvement Distribution in Masked Regions', fontweight='bold')
    ax.grid(True, alpha=0.3)

def visualize_difix_effectiveness(original: torch.Tensor | np.ndarray,
                                       difix_output: torch.Tensor | np.ndarray,
                                       gt: torch.Tensor | np.ndarray,
                                       save_path: Optional[Path] = None,
                                       title: str = "Difix Effectiveness Analysis") -> Dict:
    analyzer = InpaintingEffectivenessAnalyzer()
    e0, e1 = analyzer.compute_error_maps(original, difix_output, gt, method="combined")
    metrics = analyzer.compute_metrics(original, difix_output, gt)
    imp = analyzer.improvement_map(e0, e1)

    o = as_numpy(original)
    i = as_numpy(difix_output)
    g = as_numpy(gt)
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.8])

    # Row 0: Original images
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(o); ax.set_title("Original Rendered Image", fontsize=16); ax.axis("off")
    ax = fig.add_subplot(gs[0, 1]); ax.imshow(g); ax.set_title("Ground Truth", fontsize=16); ax.axis("off")
    ax = fig.add_subplot(gs[0, 2]); ax.imshow(i); ax.set_title("Difix Output", fontsize=16); ax.axis("off")

    # Row 1: Error maps
    ax = fig.add_subplot(gs[1, 0]); im = ax.imshow(np.squeeze(e0), cmap="hot", vmin=0, vmax=1)
    ax.set_title("Original Error Map", fontsize=16); ax.axis("off"); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[1, 1]); im = ax.imshow(np.squeeze(e1), cmap="hot", vmin=0, vmax=1)
    ax.set_title("Difix Error Map", fontsize=16); ax.axis("off"); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    vmax = float(max(abs(imp.min()), abs(imp.max()))) or 1.0
    ax = fig.add_subplot(gs[1, 2]); im = ax.imshow(imp, cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_title("Error Improvement Map", fontsize=16); ax.axis("off"); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 2: Metrics table and improvement mask
    _create_metrics_table(fig.add_subplot(gs[2, :2]), metrics)
    
    # Add binary improvement mask
    improvement_mask = (imp > 0).astype(np.float32)
    improvement_percentage = (improvement_mask.sum() / improvement_mask.size) * 100
    
    ax = fig.add_subplot(gs[2, 2])
    im = ax.imshow(improvement_mask, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Improvement Mask\n({improvement_percentage:.1f}% improved)", fontsize=14)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return metrics

# def visualize_difix_effectiveness(original: torch.Tensor | np.ndarray,
#                                        difix_output: torch.Tensor | np.ndarray,
#                                        gt: torch.Tensor | np.ndarray,
#                                        save_path: Optional[Path] = None,
#                                        title: str = "Difix Effectiveness Analysis") -> Dict:
#     analyzer = InpaintingEffectivenessAnalyzer()
#     e0, e1 = analyzer.compute_error_maps(original, difix_output, gt, method="combined")
#     metrics = analyzer.compute_metrics(original, difix_output, gt)
#     imp = analyzer.improvement_map(e0, e1)

#     o = as_numpy(original)
#     i = as_numpy(difix_output)
#     g = as_numpy(gt)
#     # m = as_numpy(mask)
    
#     fig = plt.figure(figsize=(20, 16))
#     gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.8])

#     ax = fig.add_subplot(gs[0, 0]); ax.imshow(o); ax.set_title("Original Rendered Image", fontsize=16); ax.axis("off")
#     ax = fig.add_subplot(gs[0, 1]); ax.imshow(g); ax.set_title("Ground Truth", fontsize=16); ax.axis("off")

#     # ov = o.copy(); ov[(m > 0.5), 0] = 1.0#np.clip(ov[(m > 0.5)] * [1, 0.2, 0.2], 0, 1)
#     # ax = fig.add_subplot(gs[0, 2]); ax.imshow(ov); ax.set_title("Inpainting Mask Overlay"); ax.axis("off")

#     ax = fig.add_subplot(gs[0, 2]); ax.imshow(i); ax.set_title("Difix Output", fontsize=16); ax.axis("off")

#     ax = fig.add_subplot(gs[1, 0]); im = ax.imshow(np.squeeze(e0), cmap="hot", vmin=0, vmax=1)
#     ax.set_title("Original Error Map", fontsize=16); ax.axis("off"); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

#     ax = fig.add_subplot(gs[1, 1]); im = ax.imshow(np.squeeze(e1), cmap="hot", vmin=0, vmax=1)
#     ax.set_title("Difix Error Map", fontsize=16); ax.axis("off"); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

#     vmax = float(max(abs(imp.min()), abs(imp.max()))) or 1.0
#     ax = fig.add_subplot(gs[1, 2]); im = ax.imshow(imp, cmap="RdYlGn", vmin=-vmax, vmax=vmax)
#     ax.set_title("Error Improvement Map", fontsize=16); ax.axis("off"); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


#     _create_metrics_table(fig.add_subplot(gs[2, :2]), metrics)


#     plt.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
#     plt.tight_layout()
#     if save_path:
#         Path(save_path).parent.mkdir(parents=True, exist_ok=True)
#         plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
#     plt.close(fig)
#     return metrics

def _to_hwc_np(img_t):
    if isinstance(img_t, torch.Tensor):
        if img_t.dim() == 3 and img_t.shape[0] == 3:
            return img_t.permute(1, 2, 0).detach().cpu().numpy()
        return img_t.detach().cpu().numpy()
    return img_t

def pad_to_multiple(img: Image.Image, mult: int = 8):
    w, h = img.size
    new_w = (w + mult - 1) // mult * mult
    new_h = (h + mult - 1) // mult * mult

    if (new_w, new_h) == (w, h):
        return img, (0, 0, w, h)  # no pad, crop box = full

    canvas = Image.new("RGB", (new_w, new_h))
    canvas.paste(img, (0, 0))
    # crop box to restore original size later
    crop_box = (0, 0, w, h)
    return canvas, crop_box

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train inpainting benefit predictor')
    parser.add_argument('--data_path', type=str, 
                       default="/home/dianalee/Project/3dgs/gaussian-splatting/output/LLFF/mast3r_init_opacity_decay",
                       help='Path to prepared data')
    parser.add_argument('--scenes', nargs='+', 
                       default=['fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex'],
                       help='Scenes to train on')
    parser.add_argument('--dataset', type=str, 
                       default='LLFF',
                       help='Dataset to train on')
    args = parser.parse_args()

    pipe = DifixPipeline.from_pretrained(
        "nvidia/difix_ref",
        trust_remote_code=True
    ).to("cuda")

    # pipe_no_ref = DifixPipeline.from_pretrained(
    #     "nvidia/difix",
    #     trust_remote_code=True
    # ).to("cuda")
    for scene in args.scenes:

        save_dir = os.path.join("benefit_map", "ref_image_gt", f"{args.dataset}", f"{scene}_3views")
        os.makedirs(save_dir, exist_ok=True)
        # print(f"Processing scene: {scene}")
        scene_path = Path(args.data_path) / f"{scene}_3views" / "1" / "non_train" / "ours_7000"
        features_dir = scene_path / "features"

        rgb_npy_list = sorted((features_dir / "render_images").glob("*.npy"))
        n_images = len(rgb_npy_list)

# /home/dianalee/Project/3dgs/gaussian-splatting/output/LLFF/mast3r_init_opacity_decay/fern_3views/1/train/ours_7000/renders/image001.png
        ref_scene_path = Path(args.data_path) / f"{scene}_3views" / "1" / "train" / "ours_7000" /  "gt"

        ref_image_list = sorted((ref_scene_path).glob("*.png"))
       
        main_ref_image_path = ref_image_list[0]
        main_ref_image = load_image(str(main_ref_image_path))
        main_ref_image_padded, crop_box = pad_to_multiple(main_ref_image, mult=8)
        for test_idx in tqdm(range(n_images), desc=f"Processing scene: {scene}"):

            rgb_file = rgb_npy_list[test_idx]
            base_name = rgb_file.stem
            # print(f"Processing image: {base_name}")
            
            rgb_path = features_dir / "render_images" / f"{base_name}.npy"
            rgb_png_path = features_dir / "render_images" / f"{base_name}.png"

            rend = load_npy_chw(str(rgb_path)).to(torch.float32).clamp(0,1).contiguous()

            _, H, W = rend.shape

            gt_path = features_dir / "gt" / f"{base_name}.npy"
            gt_image = load_npy_chw(str(gt_path))
           
            rendered_np = _to_hwc_np(rend)
            gt_np       = _to_hwc_np(gt_image)


            input_image = load_image(str(rgb_png_path))
            input_image_padded, crop_box = pad_to_multiple(input_image, mult=8)
            canvas_np = np.array(input_image_padded)

            prompt = "remove degradation"


            output_image = pipe(prompt, image=input_image_padded, ref_image=main_ref_image_padded, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
            # output_image = pipe_no_ref(prompt, image=input_image_padded, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
            output_np = np.array(output_image) / 255.0
            output_image_cropped = output_image.crop(crop_box)
     
            output_cropped_np = np.array(output_image_cropped) / 255.0
            # fig, axs = plt.subplots(1, 4, figsize=(20, 5))
            # plt.subplot(1, 4, 1)
            # plt.title(f"Original Image (WxH) : {input_image.size[0]}x{input_image.size[1]}")
            # plt.imshow(np.array(input_image) / 255.0)
            # plt.subplot(1, 4, 2)
            # plt.title(f"Padded Image (WxH) : {input_image_padded.size[0]}x{input_image_padded.size[1]}")
            # plt.imshow(canvas_np / 255.0)
            # plt.subplot(1, 4, 3)
            # plt.title(f"Difix output (WxH) : {output_image.size[0]}x{output_image.size[1]}")
            # plt.imshow(output_np)
           


            # plt.subplot(1, 4, 4)
            # plt.title(f"Difix output (WxH) : {output_image_cropped.size[0]}x{output_image_cropped.size[1]}")
            # plt.imshow(output_cropped_np)
            # plt.show()
            visualize_difix_effectiveness(rendered_np, output_cropped_np, gt_np, 
                                                    save_path=os.path.join(save_dir, f"{base_name}_difix_improvement.png"), 
                                                    title=f"Difix Improvement - {scene} - {base_name}")

    

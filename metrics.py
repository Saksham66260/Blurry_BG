import cv2
import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import pandas as pd

# ==================================================
# ALL SHARPNESS METRICS (Traditional + Perceptual)
# ==================================================

def laplacian_variance(image):
    """Laplacian Variance - measures edge sharpness"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


def tenengrad_sharpness(image):
    """Tenengrad - Sobel-based gradient magnitude"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.mean(np.sqrt(gx**2 + gy**2)**2)


def frequency_sharpness(image):
    """High Frequency Content - FFT-based"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    radius = min(crow, ccol)
    y, x = np.ogrid[:rows, :cols]
    mask = ((x - ccol)**2 + (y - crow)**2) > (radius * 0.6)**2
    return np.sum(magnitude * mask) / (np.sum(magnitude) + 1e-8)


def brenner_sharpness(image):
    """Brenner Gradient"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    dx = np.diff(gray.astype(np.float64), n=2, axis=1)
    return np.sum(dx**2)


def edge_density(image, threshold=50):
    """Edge Density - proportion of edge pixels"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    edges = cv2.Canny(gray, threshold, threshold * 2)
    return np.sum(edges > 0) / edges.size


def modified_laplacian(image):
    """Modified Laplacian - blur detection"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    ml = cv2.filter2D(gray.astype(np.float64), -1, kernel)
    return np.sum(np.abs(ml)) / gray.size


def perceptual_sharpness_score(image):
    """Perceptual Sharpness - weighted multi-cue metric"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Edge strength
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_score = np.percentile(edge_strength, 95)
    
    # Local contrast
    local_std = ndimage.generic_filter(gray.astype(float), np.std, size=15)
    contrast_score = np.mean(local_std)
    
    # High frequency
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    hf_score = np.std(laplacian)
    
    # Texture
    texture_score = np.std(gray)
    
    return 0.40 * edge_score + 0.30 * contrast_score + 0.20 * hf_score + 0.10 * texture_score


def local_contrast_metric(image):
    """Local Contrast - measures detail visibility"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    local_std = ndimage.generic_filter(gray.astype(float), np.std, size=15)
    return np.mean(local_std)


def edge_width_metric(image):
    """Edge Width - sharper edges are thinner"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    edges = cv2.Canny(gray, 50, 150)
    if np.sum(edges) == 0:
        return 0
    # Dilate to measure edge width
    dilated = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=1)
    width = np.sum(dilated) / (np.sum(edges) + 1e-8)
    return width


def acutance_metric(image):
    """Acutance - perceived sharpness measure"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Compute gradient
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(gx**2 + gy**2)
    
    # Weight by local average
    mean_filter = cv2.GaussianBlur(gray.astype(float), (5, 5), 1)
    acutance = np.sum(gradient * mean_filter) / (np.sum(mean_filter) + 1e-8)
    return acutance


# ==================================================
# COMPREHENSIVE METRICS COMPUTATION
# ==================================================

def compute_all_metrics(image, mask=None):
    """Compute all metrics for an image"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply mask if provided
    if mask is not None:
        if mask.shape != gray.shape:
            mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        bg_mask = (1 - mask) > 0.5
        masked_img = gray.copy()
        masked_img[~bg_mask] = masked_img[bg_mask].mean()
    else:
        masked_img = gray
    
    metrics = {
        # Traditional metrics
        'Laplacian Variance': laplacian_variance(masked_img),
        'Tenengrad': tenengrad_sharpness(masked_img),
        'Brenner Gradient': brenner_sharpness(masked_img),
        'Modified Laplacian': modified_laplacian(masked_img),
        'High Freq Content': frequency_sharpness(masked_img),
        'Edge Density': edge_density(masked_img),
        
        # Perceptual metrics
        'Perceptual Sharpness': perceptual_sharpness_score(masked_img),
        'Local Contrast': local_contrast_metric(masked_img),
        'Acutance': acutance_metric(masked_img),
        'Edge Width': edge_width_metric(masked_img),
    }
    
    return metrics


def compute_structural_metrics(original, processed):
    """Compute PSNR and SSIM"""
    if len(original.shape) == 3:
        orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    else:
        orig_gray = original
        
    if len(processed.shape) == 3:
        proc_gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
    else:
        proc_gray = processed
    
    # Ensure same size
    if orig_gray.shape != proc_gray.shape:
        proc_gray = cv2.resize(proc_gray, (orig_gray.shape[1], orig_gray.shape[0]))
    
    # PSNR
    try:
        psnr_value = psnr(orig_gray, proc_gray, data_range=255)
    except:
        psnr_value = 0
    
    # SSIM
    try:
        ssim_value = ssim(orig_gray, proc_gray, data_range=255)
    except:
        ssim_value = 0
    
    return psnr_value, ssim_value


# ==================================================
# PROFESSIONAL REPORT GENERATION
# ==================================================

def generate_comprehensive_report(original, processed, mask=None, save_prefix="metrics"):
    """Generate comprehensive metrics report for panel presentation"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE DEBLURRING PERFORMANCE METRICS")
    print("="*80)
    
    # Compute metrics
    metrics_before = compute_all_metrics(original, mask)
    metrics_after = compute_all_metrics(processed, mask)
    
    # Compute improvements
    improvements = {}
    for key in metrics_before.keys():
        before = metrics_before[key]
        after = metrics_after[key]
        
        # For Edge Width, lower is better
        if key == 'Edge Width':
            improvement = ((before - after) / (before + 1e-8)) * 100
        else:
            improvement = ((after - before) / (before + 1e-8)) * 100
        
        improvements[key] = improvement
    
    # PSNR and SSIM
    psnr_val, ssim_val = compute_structural_metrics(original, processed)
    
    # Create DataFrame for better visualization
    data = []
    for key in metrics_before.keys():
        data.append({
            'Metric': key,
            'Before': f"{metrics_before[key]:.2f}",
            'After': f"{metrics_after[key]:.2f}",
            'Improvement': f"{improvements[key]:+.1f}%"
        })
    
    df = pd.DataFrame(data)
    
    # Print traditional metrics
    print("\n📊 TRADITIONAL SHARPNESS METRICS:")
    print("-" * 80)
    traditional = ['Laplacian Variance', 'Tenengrad', 'Brenner Gradient', 
                   'Modified Laplacian', 'High Freq Content', 'Edge Density']
    
    for metric in traditional:
        before = metrics_before[metric]
        after = metrics_after[metric]
        imp = improvements[metric]
        status = "✓" if imp > 0 else "✗"
        print(f"{status} {metric:<25} {before:>12.2f} → {after:>12.2f}  ({imp:>+7.1f}%)")
    
    # Print perceptual metrics
    print("\n🎯 PERCEPTUAL QUALITY METRICS:")
    print("-" * 80)
    perceptual = ['Perceptual Sharpness', 'Local Contrast', 'Acutance', 'Edge Width']
    
    for metric in perceptual:
        before = metrics_before[metric]
        after = metrics_after[metric]
        imp = improvements[metric]
        status = "✓" if imp > 0 else "✗"
        print(f"{status} {metric:<25} {before:>12.2f} → {after:>12.2f}  ({imp:>+7.1f}%)")
    
    # Print structural metrics
    print("\n📐 STRUCTURAL SIMILARITY METRICS:")
    print("-" * 80)
    print(f"  PSNR (Peak Signal-to-Noise Ratio):     {psnr_val:.2f} dB")
    print(f"  SSIM (Structural Similarity Index):    {ssim_val:.4f}")
    
    # Overall assessment
    print("\n" + "="*80)
    print("SUMMARY STATISTICS:")
    print("="*80)
    
    positive_metrics = sum(1 for v in improvements.values() if v > 0)
    total_metrics = len(improvements)
    avg_improvement = np.mean(list(improvements.values()))
    
    print(f"  Metrics Improved:        {positive_metrics}/{total_metrics} ({positive_metrics/total_metrics*100:.1f}%)")
    print(f"  Average Improvement:     {avg_improvement:+.1f}%")
    print(f"  PSNR:                    {psnr_val:.2f} dB")
    print(f"  SSIM:                    {ssim_val:.4f}")
    
    # Key highlights
    print("\n🌟 KEY HIGHLIGHTS:")
    print("-" * 80)
    sorted_improvements = sorted(improvements.items(), key=lambda x: x[1], reverse=True)
    
    print("  Top 3 Improvements:")
    for i, (metric, imp) in enumerate(sorted_improvements[:3], 1):
        print(f"    {i}. {metric}: {imp:+.1f}%")
    
    # Performance rating
    print("\n⭐ OVERALL PERFORMANCE RATING:")
    print("-" * 80)
    
    score = 0
    if avg_improvement > 10: score += 2
    elif avg_improvement > 0: score += 1
    
    if improvements.get('Perceptual Sharpness', 0) > 10: score += 2
    elif improvements.get('Perceptual Sharpness', 0) > 0: score += 1
    
    if improvements.get('Edge Density', 0) > 20: score += 2
    elif improvements.get('Edge Density', 0) > 0: score += 1
    
    if ssim_val > 0.90: score += 1
    if psnr_val > 25: score += 1
    
    rating_map = {
        9: ("⭐⭐⭐⭐⭐", "EXCELLENT", "Outstanding deblurring performance"),
        7: ("⭐⭐⭐⭐", "VERY GOOD", "Significant quality improvement"),
        5: ("⭐⭐⭐", "GOOD", "Noticeable enhancement"),
        3: ("⭐⭐", "FAIR", "Moderate improvement"),
        0: ("⭐", "POOR", "Limited or no improvement")
    }
    
    for threshold in sorted(rating_map.keys(), reverse=True):
        if score >= threshold:
            stars, rating, desc = rating_map[threshold]
            print(f"  {stars} {rating}")
            print(f"  {desc}")
            break
    
    print("="*80)
    
    # Save detailed CSV report
    df.to_csv(f'{save_prefix}_detailed_report.csv', index=False)
    print(f"\n✓ Detailed report saved: {save_prefix}_detailed_report.csv")
    
    # Create visualization
    create_metrics_visualization(metrics_before, metrics_after, improvements, 
                                 psnr_val, ssim_val, f'{save_prefix}_chart.png')
    
    return {
        'metrics_before': metrics_before,
        'metrics_after': metrics_after,
        'improvements': improvements,
        'psnr': psnr_val,
        'ssim': ssim_val,
        'avg_improvement': avg_improvement,
        'rating': rating
    }


def create_metrics_visualization(before, after, improvements, psnr_val, ssim_val, save_path):
    """Create professional visualization for panel presentation"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Improvement Bar Chart
    ax1 = fig.add_subplot(gs[0, :])
    metrics = list(improvements.keys())
    values = list(improvements.values())
    colors = ['green' if v > 0 else 'red' for v in values]
    
    bars = ax1.barh(metrics, values, color=colors, alpha=0.7)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax1.set_xlabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Sharpness Metrics Improvement', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, values):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2, 
                f'{value:+.1f}%', ha='left' if value > 0 else 'right', 
                va='center', fontsize=9, fontweight='bold')
    
    # 2. Quality Scores
    ax2 = fig.add_subplot(gs[1, :])
    ax2.axis('off')
    
    quality_text = f"""
    STRUCTURAL QUALITY METRICS
    {'='*35}
    
    PSNR (Peak Signal-to-Noise Ratio)
    → {psnr_val:.2f} dB
    
    SSIM (Structural Similarity Index)
    → {ssim_val:.4f}
    
    Average Improvement
    → {np.mean(list(improvements.values())):+.1f}%
    
    Positive Metrics
    → {sum(1 for v in improvements.values() if v > 0)}/{len(improvements)}
    """
    
    ax2.text(0.1, 0.9, quality_text, fontsize=11, family='monospace',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Top Improvements
    ax3 = fig.add_subplot(gs[2, :])
    sorted_imp = sorted(improvements.items(), key=lambda x: abs(x[1]), reverse=True)[:6]
    metrics_top = [m for m, _ in sorted_imp]
    values_top = [v for _, v in sorted_imp]
    colors_top = ['green' if v > 0 else 'red' for v in values_top]
    
    bars = ax3.bar(metrics_top, values_top, color=colors_top, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax3.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Top 6 Metrics by Magnitude of Change', fontsize=14, fontweight='bold')
    ax3.set_xticklabels(metrics_top, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, values_top):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:+.1f}%', ha='center', va='bottom' if value > 0 else 'top',
                fontsize=10, fontweight='bold')
    
    plt.suptitle('Defocus Deblurring Performance Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved: {save_path}")
    plt.close()


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":
    print("Loading images...")
    original = cv2.imread("input.jpg")
    processed = cv2.imread("final_output.png")
    mask = cv2.imread("blur_mask.png", cv2.IMREAD_GRAYSCALE)
    
    if original is None or processed is None:
        print("Error: Could not load images")
        exit(1)
    
    if mask is not None:
        mask = mask.astype(float) / 255.0
    
    # Resize if needed
    if original.shape != processed.shape:
        print(f"⚠ Resizing processed to match original...")
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]), 
                              interpolation=cv2.INTER_CUBIC)
    
    # Generate comprehensive report
    results = generate_comprehensive_report(original, processed, mask, save_prefix="panel_metrics")
    
    print("\n✓ All reports generated successfully!")
    print("\nFiles created for panel presentation:")
    print("  1. panel_metrics_detailed_report.csv - Detailed metrics table")
    print("  2. panel_metrics_chart.png - Visual charts and graphs")
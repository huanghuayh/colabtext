import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_region_masks(model_1_output, model_2_output, index=0):
    """
    model_1_output: list like [[region_mask_f0, gt_y], ...]
    model_2_output: list like [[mask_f0, mask_2f0, mask_3f0, mask_4f0], ...]
    index: which sample to visualize
    """
    # unpack
    f0_mask, gt_y = model_1_output[index]
    masks = model_2_output[index]  # list of harmonic masks
    x = np.arange(len(f0_mask))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # ---- Row 1: fundamental vs ground truth ----
    axes[0].plot(x, gt_y, label='Ground Truth (all harmonics)', color='gray', linewidth=2, alpha=0.7)
    axes[0].plot(x, f0_mask, label='Fundamental region (Head1)', color='blue', linewidth=2)
    axes[0].set_title('Model 1 Output — Fundamental vs Ground Truth')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.4)

    # ---- Row 2: individual harmonic regions ----
    colors = ['r', 'g', 'b', 'orange']
    labels = ['f0', '2f0', '3f0', '4f0']

    for i, mask in enumerate(masks):
        axes[1].plot(x, mask, color=colors[i % len(colors)], label=labels[i], linewidth=2)

    axes[1].set_title('Model 2 Output — Individual Harmonic Regions')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.4)

    plt.xlabel('Frequency Bin')
    plt.tight_layout()
    plt.show()

def make_gaussian(zero_mask, center_idx, psd_length=1024, sigma=3):
    """Smooth Gaussian mask centered at center_idx."""
    x = np.arange(psd_length)
    gaussian = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-0.5 * ((x - center_idx) / sigma) ** 2)
    gaussian /= gaussian.max()
    final_mask = np.maximum(zero_mask, gaussian)
    return final_mask.astype(np.float32)


def make_block_mask(zero_mask, center_idx, psd_length=1024, block_width=3):
    """Binary block mask: 1 in a window around center_idx, else 0."""
    start = max(0, center_idx - block_width)
    end = min(psd_length, center_idx + block_width + 1)
    mask = zero_mask.copy()
    mask[start:end] = 1.0
    return mask.astype(np.float32)


def remake_targets(batch_y, total_num_outputs=6, mode="block", psd_length=1024, block_width=3, device='cpu'):
    """
    Builds multi-task targets for both fundamental and synthetically generated harmonics (2F0, 3F0, 4F0).
    mode: 'gaussian' or 'block'
    """
    batch_y_np = batch_y.numpy()
    model1_list, model2_list = [], []

    for b in range(batch_y_np.shape[0]):
        cur_mask = batch_y_np[b]
        f0_indices = np.where(cur_mask == 1)[0]
        if len(f0_indices) == 0:
            continue

        f0_idx = int(f0_indices[0])
        if mode == "block":
            f0_idx = int(f0_indices[0]) + block_width
        zero_mask = np.zeros(psd_length, dtype=np.float32)

        # --- compute synthetic harmonic indices ---
        harmonic_indices = [f0_idx * i for i in range(1, 5)]
        harmonic_indices = [min(idx, psd_length - 1) for idx in harmonic_indices]

        # --- make region masks ---
        if mode == "gaussian":
            f0_mask = make_gaussian(zero_mask, harmonic_indices[0], psd_length, block_width)
            f2_mask = make_gaussian(zero_mask, harmonic_indices[1], psd_length, block_width)
            f3_mask = make_gaussian(zero_mask, harmonic_indices[2], psd_length, block_width)
            f4_mask = make_gaussian(zero_mask, harmonic_indices[3], psd_length, block_width)
        elif mode == "block":
            f0_mask = make_block_mask(zero_mask, harmonic_indices[0], psd_length, block_width)
            f2_mask = make_block_mask(zero_mask, harmonic_indices[1], psd_length, block_width)
            f3_mask = make_block_mask(zero_mask, harmonic_indices[2], psd_length, block_width)
            f4_mask = make_block_mask(zero_mask, harmonic_indices[3], psd_length, block_width)
        else:
            raise ValueError("mode must be 'gaussian' or 'block'")

        # --- stack for both model heads ---
        model1_list.append(np.stack([f0_mask, cur_mask], axis=0))
        model2_list.append(np.stack([f0_mask, f2_mask, f3_mask, f4_mask], axis=0))

    model1_targets = torch.from_numpy(np.stack(model1_list)).float().to(device)
    model2_targets = torch.from_numpy(np.stack(model2_list)).float().to(device)
    return model1_targets, model2_targets
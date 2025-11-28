import numpy as np
import argparse
import os

def fourier_frequency_error(M: np.ndarray, M_hat: np.ndarray) -> float:
    """
    Calculate the Fourier Frequency Error (FFE) between two motion sequences.
    
    Args:
        M: numpy array of shape (T, 3), original motion sequence
        M_hat: numpy array of shape (T, 3), predicted motion sequence
        
    Returns:
        FFE value (float)
    """
    assert M.shape == M_hat.shape, "Input sequences must have the same shape"
    T, C = M.shape
    assert C == 3, "Input sequences must have 3 channels"
    
    # Compute FFT along time axis for each channel
    F_M = np.fft.fft(M, axis=0)
    F_M_hat = np.fft.fft(M_hat, axis=0)
    
    # Compute squared L2 norm of differences for each channel
    diff_squared = np.abs(F_M - F_M_hat) ** 2  # shape (T, 3)
    
    # Sum over time and channels
    total_error = np.sum(diff_squared)  
    
    # Normalize by N * 3 (T * 3)
    ffe = total_error / (T * 3)
    
    return ffe.real  # FFT returns complex values, but error is real

def evaluate(pred_seq, gt_seq, mouth_mask, upper_mask):
    errors = {}

    # Ensure sequences match
    pred_seq = pred_seq[:gt_seq.shape[0], :]
    gt_seq = gt_seq[:pred_seq.shape[0], :]

    # MVE & LVE
    errors['mve'] = np.linalg.norm(pred_seq - gt_seq, axis=1).mean()
    errors['lve'] = np.linalg.norm(pred_seq[:, mouth_mask] - gt_seq[:, mouth_mask], axis=1).mean()

    # MOD
    pred_offset = pred_seq[1:] - pred_seq[:-1]
    gt_offset = gt_seq[1:] - gt_seq[:-1]
    errors['mod'] = np.linalg.norm(pred_offset - gt_offset, axis=1).mean()

    # VE (Velocity Error)
    errors['ve'] = np.mean(np.linalg.norm(pred_offset - gt_offset, axis=1))

    # AE (Acceleration Error)
    pred_acc = pred_offset[1:] - pred_offset[:-1]
    gt_acc = gt_offset[1:] - gt_offset[:-1]
    errors['ae'] = np.mean(np.linalg.norm(pred_acc - gt_acc, axis=1))

    # Temporal Consistency (TC)
    pred_tc = np.mean(np.linalg.norm(pred_offset, axis=1))
    errors['tc'] = pred_tc

    # FDD & ABS FDD (upper face motion std diff)
    def motion_std(seq, mask):
        L2_dis = np.array([np.square(seq[:, v]) for v in mask])
        L2_dis = np.transpose(L2_dis, (1, 0))
        L2_dis = np.sum(L2_dis, axis=1)
        return np.std(L2_dis, axis=0)

    gt_motion_std = motion_std(gt_seq, upper_mask)
    pred_motion_std = motion_std(pred_seq, upper_mask)
    errors['fdd'] = gt_motion_std - pred_motion_std
    errors['abs_fdd'] = np.abs(gt_motion_std - pred_motion_std)

    # Fourier Frequency Error (FFE)
    # Here, we consider only the first 3 channels (e.g. x,y,z) of the sequence for FFE.
    # If your pred_seq / gt_seq have more than 3 channels, you might want to adjust accordingly.
    if pred_seq.shape[1] >= 3:
        errors['ffe'] = fourier_frequency_error(gt_seq[:, :3], pred_seq[:, :3])
    else:
        errors['ffe'] = np.nan  # Cannot compute FFE if less than 3 channels

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="BIWI")
    parser.add_argument("--pred_path", type=str, default="BIWI/result_09_27_21_23/")
    parser.add_argument("--gt_path", type=str, default="BIWI/vertices_npy/")
    args = parser.parse_args()

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", type=str, default="vocaset")
    # parser.add_argument("--pred_path", type=str, default="vocaset/result_09_27_21_20/")
    # parser.add_argument("--gt_path", type=str, default="vocaset/vertices_npy/")
    # args = parser.parse_args()

    mouth_mask = list(range(94, 114)) + list(range(146, 178)) + list(range(183, 192))
    upper_mask = [x for x in range(192) if x not in mouth_mask]

    # Accumulators
    metrics = {
        'mve': [], 'lve': [], 'mod': [],
        've': [], 'ae': [], 'tc': [],
        'fdd': [], 'abs_fdd': [], 'ffe': []
    }

    total_frames = 0
    num_seq = 0

    for file in os.listdir(args.pred_path):
        if file.endswith('.npy'):
            if args.dataset == "BIWI":
                seq_name = "_".join(os.path.basename(file).split('.')[0].split('_')[:2])
            else:
                seq_name = "_".join(os.path.basename(file).split('.')[0].split('_')[:5])

            gt_seq = np.load(os.path.join(args.gt_path, seq_name + ".npy"))
            pred_seq = np.load(os.path.join(args.pred_path, file))

            if args.dataset == "BIWI":
                gt_seq = gt_seq.reshape(-1, 3895 * 3)

            if gt_seq.shape[0] < 3 or pred_seq.shape[0] < 3:
                continue  # skip short sequences

            errs = evaluate(pred_seq, gt_seq, mouth_mask, upper_mask)

            for key in metrics:
                metrics[key].append(errs[key])

            total_frames += min(pred_seq.shape[0], gt_seq.shape[0])
            num_seq += 1

    # Print Results
    print(f'Total Frames: {total_frames}')
    print(f'Total Sequences: {num_seq}')
    print('-' * 40)
    print(f'MVE (Mean Vertex Error): {np.mean(metrics["mve"]):.4e}')
    print(f'LVE (Lip Vertex Error): {np.mean(metrics["lve"]):.4e}')
    print(f'FDD: {np.mean(metrics["fdd"]):.4e}')
    print(f'ABS FDD: {np.mean(metrics["abs_fdd"]):.4e}')
    print(f'MOD (Motion Offset Deviation): {np.mean(metrics["mod"]):.4e}')
    print(f'VE (Velocity Error): {np.mean(metrics["ve"]):.4e}')
    print(f'AE (Acceleration Error): {np.mean(metrics["ae"]):.4e}')
    print(f'TC (Temporal Consistency): {np.mean(metrics["tc"]):.4e}')
    print(f'FFE (Fourier Frequency Error): {np.mean(metrics["ffe"]):.4e}')

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Evaluate generated images with FID, IS, PSNR and SSIM and write results to a metrics file.

Usage:
  python scripts/evaluate_metrics.py --gt_dir path/to/ground_truth --gen_dir path/to/generated --out metrics.txt
"""
import argparse
import os
import sys
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
if proj_root not in sys.path:
    sys.path.insert(0,proj_root)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, required=True, help='Ground-truth images directory')
    parser.add_argument('--gen_dir', type=str, required=True, help='Generated images directory')
    parser.add_argument('--out', type=str, default='metrics.txt', help='Output metrics file path')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for IS computation')
    parser.add_argument('--splits', type=int, default=10, help='Number of splits for IS')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA for inception score (if available)')
    args = parser.parse_args()

    gt_dir = args.gt_dir
    gen_dir = args.gen_dir
    out_file = args.out

    if not os.path.isdir(gt_dir):
        print('Ground-truth directory not found:', gt_dir)
        sys.exit(1)
    if not os.path.isdir(gen_dir):
        print('Generated directory not found:', gen_dir)
        sys.exit(1)

    # Compute FID
    try:
        from cleanfid import fid as cleanfid_fid
        print('Computing FID...')
        fid_score = cleanfid_fid.compute_fid(gt_dir, gen_dir)
    except Exception as e:
        print('Failed to compute FID:', e)
        fid_score = None

    # Compute Inception Score (using models.metric.inception_score which returns mean,std)
    # --- Reliable Inception Score computation (replace original IS block) ---
    try:
        import torch
        import numpy as np
        from torchvision import transforms
        from torchvision.models import inception_v3
        from PIL import Image
        import glob
        import torch.nn.functional as F

        def compute_inception_score_from_dir(img_dir, batch_size=8, splits=10, device=None):
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # load model
            model = inception_v3(pretrained=True, transform_input=False).eval().to(device)
            # preprocessing: resize to 299 and ImageNet norm
            preprocess = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            # collect image paths
            patterns = [os.path.join(img_dir, '*' + ext) for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']]
            paths = []
            for p in patterns:
                paths.extend(glob.glob(p))
            paths = sorted(paths)
            N = len(paths)
            if N == 0:
                raise RuntimeError('No images found for IS computation in: {}'.format(img_dir))
            # adjust batch_size and splits
            batch_size = min(batch_size, max(1, N))
            splits = min(splits, max(1, N // batch_size))
            if splits == 0:
                splits = 1

            preds = []
            with torch.no_grad():
                for i in range(0, N, batch_size):
                    batch_paths = paths[i:i + batch_size]
                    imgs = []
                    for pth in batch_paths:
                        img = Image.open(pth).convert('RGB')
                        img = preprocess(img)
                        imgs.append(img)
                    batch = torch.stack(imgs, dim=0).to(device)
                    out = model(batch)  # logits (B,1000)
                    p = F.softmax(out, dim=1).cpu().numpy()
                    preds.append(p)
            preds = np.vstack(preds)  # shape (N,1000)

            # compute IS per split
            split_scores = []
            n_per_split = preds.shape[0] // splits
            eps = 1e-16
            for k in range(splits):
                start = k * n_per_split
                end = (k + 1) * n_per_split if k < splits - 1 else preds.shape[0]
                part = preds[start:end, :]
                py = np.mean(part, axis=0)
                # KL divergence for each sample
                kl = part * (np.log(part + eps) - np.log(py + eps))
                kl_sum = np.sum(kl, axis=1)
                score = np.exp(np.mean(kl_sum))
                split_scores.append(score)
            return float(np.mean(split_scores)), float(np.std(split_scores))

        print('Computing Inception Score (may take a while)...')
        device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
        is_mean, is_std = compute_inception_score_from_dir(gen_dir, batch_size=args.batch_size, splits=args.splits,
                                                           device=device)
    except Exception as e:
        print('Failed to compute Inception Score:', e)
        is_mean, is_std = None, None
    # --- end replacement ---

    # Compute PSNR / SSIM using cal_ssim_psnr from metrics.py
    try:
        from metrics import cal_ssim_psnr
        print('Computing PSNR / SSIM...')
        mpsnr, mssim = cal_ssim_psnr(gt_dir, gen_dir)
    except Exception as e:
        print('Failed to compute PSNR/SSIM:', e)
        mpsnr, mssim = None, None

    # Write results
    try:
        with open(out_file, 'w') as f:
            f.write('Results for generated dir: {}\n'.format(gen_dir))
            if fid_score is not None:
                f.write('FID: {}\n'.format(fid_score))
            else:
                f.write('FID: FAILED\n')
            if is_mean is not None:
                f.write('IS_mean: {}\n'.format(is_mean))
                f.write('IS_std: {}\n'.format(is_std))
            else:
                f.write('IS: FAILED\n')
            if mpsnr is not None:
                f.write('PSNR: {}\n'.format(mpsnr))
                f.write('SSIM: {}\n'.format(mssim))
            else:
                f.write('PSNR/SSIM: FAILED\n')
        print('Wrote metrics to', out_file)
    except Exception as e:
        print('Failed to write metrics file:', e)

if __name__ == '__main__':
    main()



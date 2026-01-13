# debug_is_stats.py
import os, glob, numpy as np, torch, torch.nn.functional as F
from torchvision import transforms
from torchvision.models import inception_v3
from PIL import Image

gen_dir = 'datasets/dunhuang/test_ref'   # <-- 改成你的生成目录
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 8
splits = 10

preprocess = transforms.Compose([
    transforms.Resize((299,299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

model = inception_v3(pretrained=True, transform_input=False).eval().to(device)

paths = sorted(glob.glob(os.path.join(gen_dir, '*.png')) + glob.glob(os.path.join(gen_dir, '*.jpg')))
N = len(paths)
print('N images =', N)
if N == 0:
    raise SystemExit('no images found')

preds = []
with torch.no_grad():
    for i in range(0, N, batch_size):
        batch_paths = paths[i:i+batch_size]
        imgs = [preprocess(Image.open(p).convert('RGB')) for p in batch_paths]
        batch = torch.stack(imgs, dim=0).to(device)
        out = model(batch)          # (B,1000)
        p = F.softmax(out, dim=1).cpu().numpy()
        preds.append(p)
preds = np.vstack(preds)  # shape (N,1000)

# per-image stats
max_probs = preds.max(axis=1)
entropies = -np.sum(preds * (np.log(preds + 1e-16)), axis=1)
print('avg max prob =', max_probs.mean(), 'median =', np.median(max_probs))
print('fraction with max>0.2 =', (max_probs>0.2).mean())
print('avg entropy =', entropies.mean(), 'median entropy=', np.median(entropies))

# marginal distribution
py = preds.mean(axis=0)
py_entropy = -np.sum(py * np.log(py + 1e-16))
print('marginal entropy (p(y)) =', py_entropy)
topk = np.argsort(py)[-10:][::-1]
print('top-10 marginal classes and probs:', list(zip(topk, py[topk])))

# IS per split
split_scores = []
n_per_split = N // splits
for k in range(splits):
    start = k * n_per_split
    end = (k+1)*n_per_split if k < splits-1 else N
    part = preds[start:end, :]
    py_part = part.mean(axis=0)
    kl = part * (np.log(part + 1e-16) - np.log(py_part + 1e-16))
    kl_sum = np.sum(kl, axis=1)
    score = np.exp(np.mean(kl_sum))
    split_scores.append(score)
print('IS splits:', split_scores)
print('IS_mean =', float(np.mean(split_scores)), 'IS_std =', float(np.std(split_scores)))

# optionally save histograms to files
import matplotlib.pyplot as plt
plt.hist(max_probs, bins=50)
plt.title('hist of per-image max prob')
plt.savefig('is_maxprob_hist.png')
plt.clf()
plt.hist(entropies, bins=50)
plt.title('hist of per-image entropy')
plt.savefig('is_entropy_hist.png')
print('Saved histograms: is_maxprob_hist.png, is_entropy_hist.png')
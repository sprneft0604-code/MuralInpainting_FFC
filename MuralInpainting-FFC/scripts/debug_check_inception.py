from torchvision.models import inception_v3
from torchvision import transforms
from PIL import Image
import torch, torch.nn.functional as F

m = inception_v3(pretrained=True, transform_input=False).eval()
t = transforms.Compose([transforms.Resize((299,299)), transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

img = t(Image.open('experiments/test_dunhuang_260103_185207/results/0/Out_1_1_6.png').convert('RGB')).unsqueeze(0)
with torch.no_grad():
    out = m(img)          # 应为 (1,1000)
    p = F.softmax(out, dim=1)
    print('out.shape=', out.shape, 'p.sum=', p.sum().item(), 'p.min/max=', p.min().item(), p.max().item())
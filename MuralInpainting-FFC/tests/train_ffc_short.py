import os
import sys
import torch
import torch.nn as nn

def run(steps=2, lr=1e-3, device="cpu"):
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)
    from models.attention_diffusion_modules.unet import UNet

    device = torch.device(device)
    b, single_c, h, w = 2, 3, 64, 64
    total_in_channels = 1 + single_c + single_c

    model = UNet(
        image_size=h,
        in_channel=total_in_channels,
        inner_channel=32,
        out_channel=3,
        res_blocks=1,
        attn_res=[1, 2, 4],
        num_heads=1,
        num_head_channels=8,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    for step in range(steps):
        model.train()
        x = torch.randn((b, single_c, h, w), device=device)
        x_noise = torch.randn((b, single_c, h, w), device=device)
        gammas = torch.ones((b,), device=device)
        target = torch.randn((b, 3, h, w), device=device)

        optim.zero_grad()
        out, att = model(x, x_noise, gammas)
        loss = loss_fn(out, target)
        loss.backward()
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.detach().norm().item()
        optim.step()

        print(f"STEP {step}: loss={loss.item():.6f}, grad_norm={grad_norm:.6f}")

if __name__ == "__main__":
    run()



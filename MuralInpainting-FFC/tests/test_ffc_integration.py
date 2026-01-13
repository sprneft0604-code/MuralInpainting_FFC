import os
import sys
import torch

def run():
    # ensure project root is on sys.path regardless of current working directory
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)
    from models.attention_diffusion_modules.unet import UNet

    device = torch.device("cpu")
    b, single_c, h, w = 2, 3, 64, 64  # attention_block expects 3-channel inputs; x and x_noise are each 3
    try:
        # UNet expects concatenated input: c = attention_block(x) has 1 channel,
        # then x and x_noise each have `single_c` channels -> total_in = 1 + single_c + single_c
        total_in_channels = 1 + single_c + single_c
        model = UNet(
            image_size=h,
            in_channel=total_in_channels,
            # inner_channel must be divisible by 32 due to GroupNorm32(32, channels)
            inner_channel=32,
            out_channel=3,
            res_blocks=1,
            attn_res=[1, 2, 4],
            num_heads=1,
            num_head_channels=8,
        )
        model.to(device)
        model.eval()
        x = torch.randn((b, single_c, h, w), device=device)
        x_noise = torch.randn((b, single_c, h, w), device=device)
        gammas = torch.ones((b,), device=device)
        with torch.no_grad():
            out, att = model(x, x_noise, gammas)
        print("FORWARD_OK", out.shape, type(att), getattr(att, "shape", None))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("FORWARD_ERROR", str(e))

if __name__ == "__main__":
    run()



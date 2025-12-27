from pipeline_difix import DifixPipeline
from diffusers.utils import load_image
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def pad_to_multiple(img: Image.Image, mult: int = 8):
    w, h = img.size
    new_w = (w + mult - 1) // mult * mult
    new_h = (h + mult - 1) // mult * mult

    if (new_w, new_h) == (w, h):
        return img, (0, 0, w, h)  # no pad, crop box = full

    canvas = Image.new("RGB", (new_w, new_h))
    canvas.paste(img, (0, 0))

    canvas_np = np.array(canvas)
    plt.subplots(1, 2, figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(np.array(img) / 255.0)
    plt.subplot(1, 2, 2)
    plt.imshow(canvas_np / 255.0)
    plt.show()
    # crop box to restore original size later
    crop_box = (0, 0, w, h)
    return canvas, crop_box

pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True).to("cuda")

image_paths = [
    "/home/dianalee/Project/3dgs/gaussian-splatting/output/LLFF/mast3r_init_opacity_decay/fern_3views/1/test/ours_7000/renders/image000.png",
    "/home/dianalee/Project/3dgs/gaussian-splatting/output/LLFF/mast3r_init_opacity_decay/fern_3views/1/test/ours_7000/renders/image008.png",
    "/home/dianalee/Project/3dgs/gaussian-splatting/output/LLFF/mast3r_init_opacity_decay/fern_3views/1/test/ours_7000/renders/image016.png",
]

prompt = "remove degradation"

with torch.no_grad():
    for p in image_paths:
        img = load_image(p)

        img_pad, crop_box = pad_to_multiple(img, mult=8)

        out = pipe(
            prompt,
            image=img_pad,
            num_inference_steps=1,
            timesteps=[199],
            guidance_scale=0.0
        ).images[0]

        # crop back to original WxH
        out = out.crop(crop_box)

        out.save(f"output_{p.split('/')[-1]}")

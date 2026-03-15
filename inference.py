import os
import argparse
import torch
import numpy as np

from PIL import Image
import face_alignment
from skimage.transform import warp

from dataset1.video_data import get_deca_tform
from models.cvthead import CVTHead


# ------------------------------
# preprocess image
# ------------------------------

def preprocess_image(img_pth, fa, device):

    img = Image.open(img_pth).convert("RGB")
    img_npy = np.array(img)

    # detect landmark
    landmark = fa.get_landmarks(img_npy)[0]

    # deca transform
    tform = get_deca_tform(landmark)

    img_256 = np.array(Image.fromarray(img_npy).resize((256,256)))
    img_256 = img_256 / 255.

    crop = warp(
        img_256,
        tform.inverse,
        output_shape=(224,224)
    )

    img_tensor = torch.from_numpy(img_256).float()
    img_tensor = img_tensor.permute(2,0,1)
    img_tensor = (img_tensor - 0.5) / 0.5

    crop = torch.tensor(crop).float()
    crop = crop.permute(2,0,1)

    # FIX ở đây
    tform = torch.tensor(tform.params).float()

    img_tensor = img_tensor.unsqueeze(0).to(device)
    crop = crop.unsqueeze(0).to(device)
    tform = tform.unsqueeze(0).to(device)

    return img_tensor, crop, tform

# ------------------------------
# save gif
# ------------------------------

def save_gif(frames, path):

    frames[0].save(
        path,
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=200,
        loop=0
    )


# ------------------------------
# convert tensor to image
# ------------------------------

def tensor_to_image(pred_img, pred_mask):

    # normalize image
    pred_img = 0.5 * (pred_img + 1)

    # remove batch
    pred_img = pred_img.squeeze(0)
    pred_mask = pred_mask.squeeze(0)

    # convert
    pred_img = pred_img.permute(1,2,0).detach().cpu().numpy()
    pred_mask = pred_mask.detach().cpu().numpy()

    # ensure mask shape (H,W,1)
    if pred_mask.ndim == 3:
        pred_mask = pred_mask[0]

    pred_mask = pred_mask[..., None]

    # apply mask
    background = np.ones_like(pred_img)

    img = pred_img * pred_mask + background * (1 - pred_mask)

    img = (img * 255).astype(np.uint8)

    return Image.fromarray(img)


# ------------------------------
# flame coefficient demo
# ------------------------------

def driven_by_flame_coefs(model, src_pth, out_dir, device):

    os.makedirs(out_dir, exist_ok=True)

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device=device
    )

    src_img, src_crop, src_tform = preprocess_image(
        src_pth,
        fa,
        device
    )

    # ---------------- SHAPE ----------------

    frames=[]

    with torch.no_grad():

        pose = torch.zeros(1,6).to(device)

        for i in range(10):

            shape = torch.zeros(1,100).to(device)

            shape[0,0] = 2*i/10

            outputs = model.flame_coef_generation(
                src_crop,
                src_img,
                src_tform,
                shape=shape,
                pose=pose
            )

            frame = tensor_to_image(
                outputs["pred_drv_img"],
                outputs["pred_drv_mask"]
            )

            frames.append(frame)

    save_gif(frames, os.path.join(out_dir,"shape.gif"))


    # ---------------- EXP ----------------

    frames=[]

    with torch.no_grad():

        pose = torch.zeros(1,6).to(device)

        for i in range(10):

            exp = torch.zeros(1,100).to(device)

            exp[0,0] = 2*i/10

            outputs = model.flame_coef_generation(
                src_crop,
                src_img,
                src_tform,
                exp=exp,
                pose=pose
            )

            frame = tensor_to_image(
                outputs["pred_drv_img"],
                outputs["pred_drv_mask"]
            )

            frames.append(frame)

    save_gif(frames, os.path.join(out_dir,"exp.gif"))


    # ---------------- POSE ----------------

    frames=[]

    with torch.no_grad():

        for i in range(12):

            pose = torch.zeros(1,6).to(device)

            pose[0,1] = -np.pi/4 + i*np.pi/24

            outputs = model.flame_coef_generation(
                src_crop,
                src_img,
                src_tform,
                pose=pose
            )

            frame = tensor_to_image(
                outputs["pred_drv_img"],
                outputs["pred_drv_mask"]
            )

            frames.append(frame)

    save_gif(frames, os.path.join(out_dir,"pose.gif"))


    # ---------------- JAW ----------------

    frames=[]

    with torch.no_grad():

        for i in range(12):

            pose = torch.zeros(1,6).to(device)

            pose[0,3] = 0.5*i/12

            outputs = model.flame_coef_generation(
                src_crop,
                src_img,
                src_tform,
                pose=pose
            )

            frame = tensor_to_image(
                outputs["pred_drv_img"],
                outputs["pred_drv_mask"]
            )

            frames.append(frame)

    save_gif(frames, os.path.join(out_dir,"jaw.gif"))


# ------------------------------
# main
# ------------------------------

def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:",device)

    model = CVTHead()

    model = model.to(device)

    print("Loading checkpoint...")

    ckpt = torch.load(args.ckpt_pth, map_location="cpu")

    if "model" in ckpt:
        ckpt = ckpt["model"]

    model.load_state_dict(ckpt, strict=False)

    model.eval()

    driven_by_flame_coefs(
        model,
        args.src_pth,
        args.out_dir,
        device
    )

    print("Done!")


# ------------------------------
# run
# ------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src_pth",
        type=str,
        default="examples/1.png"
    )

    parser.add_argument(
        "--ckpt_pth",
        type=str,
        default="data/cvthead.pt"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs"
    )

    args = parser.parse_args()

main(args)
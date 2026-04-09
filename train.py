import os
import glob
import random
import argparse
import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.cvthead import CVTHead

# dataset (load preprocessed npy)
class FaceDataset(Dataset):

    def __init__(self, data_dir):

        self.img_list = sorted(glob.glob(os.path.join(data_dir, "*_img.npy")))

        print("Dataset size:", len(self.img_list))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        src_path = self.img_list[idx]
        drv_path = random.choice(self.img_list)

        # load src
        src_img = torch.from_numpy(np.load(src_path)).float().permute(2,0,1)
        src_crop = torch.from_numpy(np.load(src_path.replace("_img", "_crop"))).float().permute(2,0,1)
        src_tform = torch.from_numpy(np.load(src_path.replace("_img", "_tform"))).float()

        # load drv
        drv_img = torch.from_numpy(np.load(drv_path)).float().permute(2,0,1)
        drv_crop = torch.from_numpy(np.load(drv_path.replace("_img", "_crop"))).float().permute(2,0,1)
        drv_tform = torch.from_numpy(np.load(drv_path.replace("_img", "_tform"))).float()

        return src_img, drv_img, src_crop, drv_crop, src_tform, drv_tform


def train_epoch(model, loader, optimizer, device):

    model.train()
    total_loss = 0

    pbar = tqdm(loader)

    for batch in pbar:

        src_img, drv_img, src_crop, drv_crop, src_tform, drv_tform = batch

        src_img = src_img.to(device)
        drv_img = drv_img.to(device)
        src_crop = src_crop.to(device)
        drv_crop = drv_crop.to(device)
        src_tform = src_tform.to(device)
        drv_tform = drv_tform.to(device)

        outputs = model(
            src_crop,
            drv_crop,
            src_img,
            drv_img,
            src_tform,
            drv_tform,
            is_train=True,
            is_cross_id=True
        )

        pred = outputs["pred_drv_img"]

        loss = F.l1_loss(pred, drv_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_description(f"loss {loss.item():.4f}")

    return total_loss / len(loader)


# save checkpoint
def save_ckpt(model, optimizer, epoch, path):

    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }, path)

def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    dataset = FaceDataset(args.train_dir)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0   # FIX WINDOWS
    )

    model = CVTHead().to(device)

    # load pretrained (QUAN TRỌNG)
    if args.pretrained:
        print("Loading pretrained...")
        ckpt = torch.load(args.pretrained, map_location="cpu")
        if "model" in ckpt:
            ckpt = ckpt["model"]
        model.load_state_dict(ckpt, strict=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    os.makedirs(args.output, exist_ok=True)

    for epoch in range(args.epochs):

        print(f"\nEpoch {epoch}")

        loss = train_epoch(
            model,
            loader,
            optimizer,
            device
        )

        print("Epoch loss:", loss)

        if epoch % 5 == 0:
            save_ckpt(
                model,
                optimizer,
                epoch,
                os.path.join(args.output, f"ckpt_{epoch}.pt")
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", default="dataset_preprocessed")
    parser.add_argument("--output", default="checkpoints")

    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)

    args = parser.parse_args()

    main(args)
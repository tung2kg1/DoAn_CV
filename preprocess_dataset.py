import os
import glob
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import face_alignment
from skimage.transform import warp

from dataset.video_data import get_deca_tform


# preprocess image
def process_image(img_path, fa):

    try:
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        # detect landmark
        landmarks = fa.get_landmarks(img_np)
        if landmarks is None:
            return None

        landmark = landmarks[0]

        # get transform
        tform = get_deca_tform(landmark)

        # resize 256
        img_256 = np.array(Image.fromarray(img_np).resize((256, 256)))
        img_256 = img_256 / 255.0   # [0,1]

        # crop 224 bằng warp (DECA)
        crop_224 = warp(
            img_256,
            tform.inverse,
            output_shape=(224, 224)
        )

        return img_256, crop_224, tform.params

    except Exception as e:
        print(f"Error: {img_path}")
        return None


def main(args):

    os.makedirs(args.out_dir, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))

    if args.max_images:
        img_paths = img_paths[:args.max_images]

    print("Total images:", len(img_paths))

    # face alignment (chỉ tạo 1 lần)
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device=args.device
    )

    save_count = 0

    for img_path in tqdm(img_paths):

        result = process_image(img_path, fa)

        if result is None:
            continue

        img_256, crop_224, tform = result

        name = os.path.splitext(os.path.basename(img_path))[0]

        # save numpy files
        np.save(os.path.join(args.out_dir, f"{name}_img.npy"), img_256)
        np.save(os.path.join(args.out_dir, f"{name}_crop.npy"), crop_224)
        np.save(os.path.join(args.out_dir, f"{name}_tform.npy"), tform)

        save_count += 1

    print(f"\nDone! Saved {save_count} samples.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, default="dataset")
    parser.add_argument("--out_dir", type=str, default="dataset_preprocessed")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_images", type=int, default=None)

    args = parser.parse_args()

    main(args)
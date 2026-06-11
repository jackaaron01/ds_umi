#!/usr/bin/env python3
"""
Extract visual features from rendered camera images using pretrained ResNet.

Reads HDF5 files with sensors/camera/rgb, runs each frame through ResNet18,
and stores 512-dim feature vectors as observation.image_features.

Usage:
    python3 extract_image_features.py --input data/goal_img_dataset --output data/goal_img_feat_dataset
"""
import os, sys, argparse, time
import numpy as np
import h5py
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

sys.path.insert(0, "/workspace/umi")


class ImageFeatureExtractor:
    """Extract features from rendered images using pretrained ResNet18."""

    def __init__(self, device: torch.device, feature_dim: int = 512):
        # Load pretrained ResNet18, remove final FC layer
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self._backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
        self._backbone.to(device)
        self._backbone.eval()
        self._device = device
        self._feature_dim = feature_dim

        # ImageNet normalization
        self._transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, images: np.ndarray) -> np.ndarray:
        """Extract features from batch of images.

        Args:
            images: (N, H, W, 3) uint8 array

        Returns:
            features: (N, feature_dim) float32 array
        """
        features = []
        batch_size = 64

        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            # Convert to PIL images and transform
            tensors = []
            for img in batch:
                pil = Image.fromarray(img.astype(np.uint8))
                t = self._transform(pil)
                tensors.append(t)
            batch_t = torch.stack(tensors).to(self._device)

            feat = self._backbone(batch_t)
            feat = feat.squeeze(-1).squeeze(-1)  # (B, 512)
            features.append(feat.cpu().numpy().astype(np.float32))

        return np.concatenate(features, axis=0)

    @property
    def feature_dim(self):
        return self._feature_dim


def main():
    parser = argparse.ArgumentParser(description="Extract image features from HDF5 episodes")
    parser.add_argument("--input", "-i", required=True,
                        help="Input directory with HDF5 episode files")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory for feature-enhanced HDF5 files")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    extractor = ImageFeatureExtractor(device)

    h5_files = sorted([f for f in os.listdir(args.input) if f.endswith(".h5")])
    if not h5_files:
        print("No HDF5 files found!")
        return

    total_frames = 0
    total_time = 0

    for fname in h5_files:
        src_path = os.path.join(args.input, fname)
        dst_path = os.path.join(args.output, fname)

        with h5py.File(src_path, "r") as src:
            eps = [k for k in src.keys() if k.startswith("episode_")]
            if not eps:
                print(f"  {fname}: no episode group, skipping")
                continue
            ep_name = eps[0]
            ep = src[ep_name]

            if "sensors/camera/rgb" not in ep:
                print(f"  {fname}: no images, copying as-is")
                import shutil
                shutil.copy2(src_path, dst_path)
                continue

            images = ep["sensors/camera/rgb"][:]
            n_frames = len(images)
            print(f"  {fname}: {n_frames} frames, extracting features...", end="", flush=True)

            t0 = time.time()
            features = extractor.extract(images)
            dt = time.time() - t0
            total_frames += n_frames
            total_time += dt

            print(f" {dt:.1f}s ({n_frames/dt:.0f} fps), feat shape={features.shape}")

            # Copy everything from source, add image_features
            import shutil
            shutil.copy2(src_path, dst_path)

            with h5py.File(dst_path, "r+") as dst:
                dst_ep = dst[ep_name]
                if "observation/image_features" in dst_ep:
                    del dst_ep["observation/image_features"]
                dst_ep.create_dataset("observation/image_features",
                                      data=features, compression="gzip")

    print(f"\nDone: {len(h5_files)} episodes, {total_frames} frames "
          f"in {total_time:.1f}s ({total_frames/total_time:.0f} fps)")


if __name__ == "__main__":
    main()

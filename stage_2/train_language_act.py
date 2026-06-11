#!/usr/bin/env python3
"""
Language-conditioned ACT — the practical "VLA" for our hardware.

Uses sentence-transformers to encode task descriptions into embeddings,
then trains ACT with (state + language_embedding) → action.

This is a VLA-like system scaled to fit RTX 4060 8GB:
  V: Camera images (via offscreen rendering + ResNet)
  L: Language descriptions (via sentence-transformers)
  A: Joint actions (ACT policy)

Usage:
    python3 train_language_act.py --data data/goal_dataset_v3
"""
import os, sys, time, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "/workspace/umi")

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType, PolicyFeature
from sentence_transformers import SentenceTransformer

# ═══════════════════════════════════════════════════════════════
# Task descriptions (matching our 8 goal positions)
# ═══════════════════════════════════════════════════════════════
TASK_DESCRIPTIONS = [
    "return to the home position",
    "reach forward and to the right at shoulder height",
    "reach to the left side with arm extended",
    "reach forward at middle height",
    "extend the arm forward and downward",
    "fold the arm into a compact pose to the left",
    "reach far forward and upward to the right",
    "extend the arm far to the left and downward",
]


class LanguageConditionedDataset(Dataset):
    """Dataset that adds language embeddings to state+goal observations."""

    def __init__(self, data_dir: str, chunk_size: int = 100,
                 encoder_model: str = "all-MiniLM-L6-v2"):
        self.chunk_size = chunk_size

        # Load data
        dfs = []
        data_path = os.path.join(data_dir, "data")
        for root, _, files in os.walk(data_path):
            for f in sorted(files):
                if f.endswith(".parquet"):
                    dfs.append(pd.read_parquet(os.path.join(root, f)))
        self.df = pd.concat(dfs, ignore_index=True)

        # Load sentence encoder
        print(f"Loading sentence encoder: {encoder_model}...")
        self._encoder = SentenceTransformer(encoder_model)
        self._lang_dim = self._encoder.get_sentence_embedding_dimension()
        print(f"  Language embedding dim: {self._lang_dim}")

        # Pre-compute task embeddings
        self._task_embeddings = {}
        for i, desc in enumerate(TASK_DESCRIPTIONS):
            emb = self._encoder.encode(desc, convert_to_numpy=True).astype(np.float32)
            self._task_embeddings[i] = emb

        self._has_goal = "observation.goal_position" in self.df.columns

        self.episodes = []
        self.samples = []
        for ep_idx, group in self.df.groupby("episode_index"):
            obs = np.vstack(group["observation.joint_position"].values).astype(np.float32)
            act = np.vstack(group["action.joint_position"].values).astype(np.float32)
            goal = None
            if self._has_goal:
                goal = np.vstack(group["observation.goal_position"].values).astype(np.float32)
            # Determine task from episode index (8 goals, cycled)
            task_idx = ep_idx % len(TASK_DESCRIPTIONS)
            ep_len = len(act)
            list_idx = len(self.episodes)
            self.episodes.append((obs, act, goal, task_idx))
            for f in range(max(1, ep_len - chunk_size)):
                self.samples.append((list_idx, f))

        print(f"Lang Dataset: {len(self.episodes)} eps, {len(self.samples)} samples, "
              f"lang_dim={self._lang_dim}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_idx, frame_idx = self.samples[idx]
        obs_arr, act_arr, goal_arr, task_idx = self.episodes[ep_idx]

        obs = torch.from_numpy(obs_arr[frame_idx])
        lang_emb = torch.from_numpy(self._task_embeddings[task_idx])

        end = min(frame_idx + self.chunk_size, len(act_arr))
        action = np.zeros((self.chunk_size, 6), dtype=np.float32)
        n_valid = end - frame_idx
        if n_valid > 0:
            action[:n_valid] = act_arr[frame_idx:end]

        result = [obs, lang_emb]
        if self._has_goal and goal_arr is not None:
            result.append(torch.from_numpy(goal_arr[frame_idx]))
        result.append(torch.from_numpy(action))
        return tuple(result)

    @property
    def lang_dim(self):
        return self._lang_dim


def main():
    parser = argparse.ArgumentParser(description="Train language-conditioned ACT")
    parser.add_argument("--data", default="/workspace/umi/data/goal_la_big_dataset_v3")
    parser.add_argument("--output", default="/workspace/umi/outputs/act_language")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--dim-model", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--no-goal", action="store_true",
                        help="Train WITHOUT goal_position (language-only)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset = LanguageConditionedDataset(args.data, chunk_size=args.chunk_size)
    has_goal = dataset._has_goal and not args.no_goal
    if args.no_goal:
        dataset._has_goal = False
        print("Training WITHOUT goal_position (language-only conditioning)")
    lang_dim = dataset.lang_dim

    input_features = {
        "observation.environment_state": PolicyFeature(shape=[6], type=FeatureType.ENV),
        "observation.language_embedding": PolicyFeature(shape=[lang_dim], type=FeatureType.ENV),
    }
    if has_goal:
        input_features["observation.goal_position"] = PolicyFeature(shape=[6], type=FeatureType.ENV)

    cfg = ACTConfig(
        chunk_size=args.chunk_size, n_action_steps=args.chunk_size, n_obs_steps=1,
        input_features=input_features,
        output_features={"action": PolicyFeature(shape=[6], type=FeatureType.ACTION)},
        dim_model=args.dim_model, n_heads=8, n_encoder_layers=4, n_decoder_layers=1,
        dim_feedforward=3200, dropout=0.1, use_vae=False,
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2 if device.type == "cuda" else 0,
                        pin_memory=(device.type == "cuda"))

    model = ACTPolicy(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params, lang_dim={lang_dim}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    model.train()
    losses, best_loss, t0 = [], float("inf"), time.time()
    it = iter(loader)

    for step in range(args.steps):
        try:
            bd = next(it)
        except StopIteration:
            it = iter(loader)
            bd = next(it)

        idx = 0
        obs = bd[idx].to(device); idx += 1
        lang = bd[idx].to(device); idx += 1
        goal = bd[idx].to(device) if has_goal else None; idx += int(has_goal)
        action = bd[idx].to(device)

        opt.zero_grad()
        batch = {
            "observation.environment_state": obs,
            "observation.language_embedding": lang,
            "observation.state": obs.unsqueeze(1),
            "action": action,
            "action_is_pad": torch.zeros(obs.size(0), args.chunk_size, dtype=torch.bool, device=device),
        }
        if has_goal:
            batch["observation.goal_position"] = goal

        loss, _ = model.forward(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
        losses.append(loss.item())

        if (step + 1) % 500 == 0:
            a = np.mean(losses[-500:])
            print(f"  Step {step+1:5d}/{args.steps}: loss={loss.item():.4f}, "
                  f"avg500={a:.4f}, {(step+1)/(time.time()-t0):.0f} steps/s")

        if (step + 1) % args.save_every == 0:
            al = np.mean(losses[-500:])
            if al < best_loss:
                best_loss = al
                torch.save({"step": step+1, "model_state_dict": model.state_dict(),
                             "cfg": cfg, "lang_dim": lang_dim,
                             "task_descriptions": TASK_DESCRIPTIONS},
                           os.path.join(args.output, "best.pt"))
                print(f"  -> Best ({al:.4f})")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"Loss: first500={np.mean(losses[:500]):.4f}, last500={np.mean(losses[-500:]):.4f}")
    torch.save({"step": args.steps, "model_state_dict": model.state_dict(),
                 "cfg": cfg, "lang_dim": lang_dim,
                 "task_descriptions": TASK_DESCRIPTIONS},
               os.path.join(args.output, "final.pt"))


if __name__ == "__main__":
    main()

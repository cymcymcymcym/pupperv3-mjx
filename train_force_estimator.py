#!/usr/bin/env python3
"""
Train a Flax MLP to estimate external force vectors from locomotion observations.

The trainer loads datasets produced by `collect_force_data.py`, standardizes the inputs,
and fits a two-layer MLP with dropout and L2 regularization. The best checkpoint (based on
validation loss) is saved along with an exported JSON model compatible with deployment.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import flax.linen as nn
from flax.training import train_state
import jax
from jax import numpy as jnp
import numpy as np
import optax
from orbax import checkpoint as ocp

from pupperv3_mjx.force_estimator import ForceEstimator


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout_rate: float = 0.1
    batch_size: int = 512
    num_epochs: int = 100
    patience: int = 10
    val_split: float = 0.1
    min_delta: float = 1e-4


class TrainState(train_state.TrainState):
    model: nn.Module


def prepare_data(dataset_path: Path, val_split: float, seed: int):
    data = np.load(dataset_path)
    observations = data["observations"].astype(np.float32)
    forces = data["forces"].astype(np.float32)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(observations))
    observations = observations[perm]
    forces = forces[perm]

    val_size = int(len(observations) * val_split)
    val_obs = observations[:val_size]
    val_forces = forces[:val_size]
    train_obs = observations[val_size:]
    train_forces = forces[val_size:]

    obs_mean = train_obs.mean(axis=0, keepdims=True)
    obs_std = train_obs.std(axis=0, keepdims=True) + 1e-6

    train_obs_norm = (train_obs - obs_mean) / obs_std
    val_obs_norm = (val_obs - obs_mean) / obs_std

    return (
        train_obs_norm,
        train_forces,
        val_obs_norm,
        val_forces,
        obs_mean.squeeze().astype(np.float32),
        obs_std.squeeze().astype(np.float32),
    )


def create_train_state(rng, model: nn.Module, learning_rate, weight_decay):
    dummy_input = jnp.zeros((1, model.hidden_size * 0 + 720))  # placeholder; overwritten later
    params = model.init(rng, dummy_input, train=True)["params"]
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx, model=model)


def train_epoch(state, train_obs, train_forces, batch_size, rng):
    num_samples = train_obs.shape[0]
    perms = np.random.permutation(num_samples)

    total_loss = 0.0
    num_batches = 0
    dropout_rng = rng

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_idx = perms[start:end]
        batch_obs = jnp.array(train_obs[batch_idx])
        batch_force = jnp.array(train_forces[batch_idx])

        dropout_rng, subrng = jax.random.split(dropout_rng)

        def loss_fn(params):
            preds = state.model.apply(
                {"params": params},
                batch_obs,
                train=True,
                rngs={"dropout": subrng},
            )
            loss = jnp.mean((preds - batch_force) ** 2)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        total_loss += float(loss)
        num_batches += 1

    return state, total_loss / max(1, num_batches), dropout_rng


@jax.jit
def eval_step(state, obs, forces):
    preds = state.model.apply({"params": state.params}, obs, train=False)
    loss = jnp.mean((preds - forces) ** 2)
    return loss


def export_model(params, model: ForceEstimator, input_mean, input_std, export_path: Path):
    layers = []
    params_dict = params
    for name, layer_params in params_dict.items():
        if "kernel" not in layer_params:
            continue
        kernel = np.array(layer_params["kernel"])
        bias = np.array(layer_params["bias"])
        activation = "elu" if "Dense" in name and "2" not in name else "tanh"
        layers.append(
            {
                "type": "dense",
                "activation": activation,
                "shape": [None, int(bias.shape[0])],
                "weights": [kernel.tolist(), bias.tolist()],
            }
        )

    export_dict = {
        "input_mean": input_mean.tolist(),
        "input_std": input_std.tolist(),
        "layers": layers[:-1] + [
            {
                "type": "dense",
                "activation": "identity",
                "shape": [None, 3],
                "weights": layers[-1]["weights"],
            }
        ],
    }

    with open(export_path, "w") as f:
        json.dump(export_dict, f, indent=2)


def train_force_estimator(args):
    (
        train_obs,
        train_forces,
        val_obs,
        val_forces,
        input_mean,
        input_std,
    ) = prepare_data(args.dataset, args.val_split, args.seed)

    obs_dim = train_obs.shape[1]
    model = ForceEstimator(hidden_size=args.hidden_size, dropout_rate=args.dropout_rate)

    rng = jax.random.PRNGKey(args.seed)
    init_rng, train_rng = jax.random.split(rng)

    dummy_input = jnp.zeros((1, obs_dim))
    params = model.init(init_rng, dummy_input, train=True)["params"]
    tx = optax.adamw(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx, model=model)

    best_val_loss = float("inf")
    patience_counter = 0
    dropout_rng = train_rng

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "force_estimator_checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    export_path = output_dir / "force_estimator.json"
    best_ckpt_path = ckpt_dir / "best"
    checkpointer = ocp.PyTreeCheckpointer()

    for epoch in range(1, args.num_epochs + 1):
        state, train_loss, dropout_rng = train_epoch(
            state, train_obs, train_forces, args.batch_size, dropout_rng
        )

        val_loss = float(
            eval_step(state, jnp.array(val_obs), jnp.array(val_forces))
        )

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

        if val_loss + args.min_delta < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpointer.save(
                best_ckpt_path,
                {"params": state.params, "input_mean": input_mean, "input_std": input_std},
                force=True,
            )
            export_model(state.params, model, input_mean, input_std, export_path)
            print(f"  Saved new best checkpoint (val_loss={val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    print(f"Training completed. Best val loss: {best_val_loss:.6f}")
    print(f"Exported model to {export_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train force estimator MLP.")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to dataset .npz file produced by collect_force_data.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("force_estimator_training"),
        help="Directory to store checkpoints and exported model.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    train_force_estimator(args)


if __name__ == "__main__":
    main()




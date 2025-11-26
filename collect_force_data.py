#!/usr/bin/env python3
"""
Collect (observation, force_vector) pairs by rolling out the trained locomotion policy.

The rollout configuration mirrors `verify_force_with_policy.py` but records data to disk
for supervising a force estimator. Episodes terminate early using the environment's
existing termination thresholds. When an episode ends the policy is reset and data
collection continues until the requested number of samples is gathered.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Tuple

os.environ.setdefault("MUJOCO_GL", "egl")

import jax
from jax import numpy as jp
import numpy as np

from pupperv3_mjx import config, environment, utils


def load_policy_from_json(policy_path: Path):
    """Load dense layers from exported policy JSON."""
    with open(policy_path) as f:
        policy_dict = json.load(f)

    layers = policy_dict["layers"]
    weights = []
    for layer in layers:
        kernel = jp.array(layer["weights"][0])
        bias = jp.array(layer["weights"][1])
        weights.append((kernel, bias))

    activation_fn = utils.activation_fn_map(layers[0]["activation"])

    def policy_fn(obs: jp.ndarray, rng: jp.ndarray) -> Tuple[jp.ndarray, dict]:
        x = obs
        for i, (kernel, bias) in enumerate(weights):
            x = x @ kernel + bias
            if i < len(weights) - 1:
                x = activation_fn(x)
            else:
                # Final layer uses tanh
                x = jp.tanh(x)
        return x, {}

    return policy_fn, policy_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect force estimator supervision data.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("force_estimator_data"),
        help="Directory to store collected dataset.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200_000,
        help="Number of (observation, force) samples to collect.",
    )
    parser.add_argument(
        "--policy-json",
        type=Path,
        default=Path("output_wobbly-sun-36-20251120T103459Z-1-001/output_wobbly-sun-36/policy.json"),
        help="Path to exported policy JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for rollout sampling.",
    )
    parser.add_argument(
        "--progress-steps",
        type=int,
        default=10,
        help="Number of progress updates (e.g. 10 => every 10% of samples).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel worker processes to use for collection.",
    )
    return parser.parse_args()


def collect_dataset(
    policy_path: Path,
    xml_path: Path,
    output_dir: Path,
    num_samples: int,
    seed: int,
    progress_steps: int,
    dataset_name: str,
) -> Path:
    policy_fn, policy_dict = load_policy_from_json(policy_path)
    jit_policy = jax.jit(policy_fn)

    reward_config = config.get_config()
    env = environment.PupperV3Env(
        path=str(xml_path),
        reward_config=reward_config,
        action_scale=policy_dict["action_scale"],
        observation_history=policy_dict["observation_history"],
        dof_damping=policy_dict["kd"],
        position_control_kp=policy_dict["kp"],
        joint_lower_limits=policy_dict["joint_lower_limits"],
        joint_upper_limits=policy_dict["joint_upper_limits"],
        default_pose=jp.array(policy_dict["default_joint_pos"]),
        use_imu=policy_dict["use_imu"],
        force_probability=0.8,
        force_duration_range=jp.array([40, 120]),
        force_magnitude_range=jp.array([0.8, 1.2]),
    )

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    vx, vy, wz = 0.5, 0.4, 1.5
    command_seq = jp.array(
        [
            [0.0, 0.0, 0.0],
            [vx, 0.0, 0.0],
            [-vx, 0.0, 0.0],
            [0.0, vy, 0.0],
            [0.0, -vy, 0.0],
            [0.0, 0.0, wz],
            [0.0, 0.0, -wz],
        ]
    )
    command_change_interval = 100

    rng = jax.random.PRNGKey(seed)
    state = jit_reset(rng)
    state.info["command"] = command_seq[0]

    observations = []
    forces = []
    steps = 0

    checkpoints = []
    if progress_steps > 0:
        increments = np.linspace(0, num_samples, progress_steps + 1, dtype=int)[1:]
        checkpoints = list(dict.fromkeys(increments))
    next_checkpoint_idx = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    def save_partial(filename: Path, meta_extra: dict):
        if not observations:
            return
        obs_np = np.asarray(observations, dtype=np.float32)
        forces_np = np.asarray(forces, dtype=np.float32)
        metadata = {
            "obs_dim": obs_np.shape[1],
            "num_samples": int(obs_np.shape[0]),
            "force_mean": forces_np.mean(axis=0).tolist(),
            "force_std": forces_np.std(axis=0).tolist(),
            "command_change_interval": command_change_interval,
            "force_probability": 0.8,
            "force_duration_range": [40, 120],
            "force_magnitude_range": [0.8, 1.2],
        }
        metadata.update(meta_extra)
        np.savez_compressed(
            filename,
            observations=obs_np,
            forces=forces_np,
            metadata=json.dumps(metadata),
        )

    print(f"[seed={seed}] Collecting {num_samples} samples...")
    start_time = time.time()
    while len(observations) < num_samples:
        steps += 1
        rng, act_rng = jax.random.split(rng)

        cmd_idx = (steps // command_change_interval) % len(command_seq)
        state.info["command"] = command_seq[cmd_idx]

        action, _ = jit_policy(state.obs, act_rng)
        state = jit_step(state, action)

        obs_np = np.asarray(state.obs)
        force_np = np.asarray(state.info["force_current_vector"])
        if np.linalg.norm(force_np) >= 1e-6:
            observations.append(obs_np)
            forces.append(force_np)

        if state.done:
            state = jit_reset(rng)
            state.info["command"] = command_seq[0]

        if (
            checkpoints
            and next_checkpoint_idx < len(checkpoints)
            and len(observations) >= checkpoints[next_checkpoint_idx]
        ):
            count = len(observations)
            pct = 100.0 * count / num_samples
            print(f"[seed={seed}]   {count}/{num_samples} samples ({pct:.1f}%)")
            partial_path = output_dir / f"{dataset_name}_partial_{count}.npz"
            save_partial(partial_path, {"partial": True})
            next_checkpoint_idx += 1
            elapsed = time.time() - start_time
            rate = count / elapsed if elapsed > 0 else 0.0
            remaining = (num_samples - count) / rate if rate > 0 else float("inf")
            print(
                f"[seed={seed}]     elapsed={elapsed:.1f}s | samples/sec={rate:.1f} | eta={remaining:.1f}s"
            )

    observations_np = np.asarray(observations[:num_samples])
    forces_np = np.asarray(forces[:num_samples])

    output_path = output_dir / f"{dataset_name}.npz"
    metadata = {
        "obs_dim": observations_np.shape[1],
        "num_samples": int(observations_np.shape[0]),
        "force_mean": forces_np.mean(axis=0).tolist(),
        "force_std": forces_np.std(axis=0).tolist(),
        "command_change_interval": command_change_interval,
        "force_probability": 0.8,
        "force_duration_range": [40, 120],
        "force_magnitude_range": [0.8, 1.2],
    }

    np.savez_compressed(
        output_path,
        observations=observations_np,
        forces=forces_np,
        metadata=json.dumps(metadata),
    )

    total_elapsed = time.time() - start_time
    samples_per_sec = (
        observations_np.shape[0] / total_elapsed if total_elapsed > 0 else 0.0
    )
    print(
        f"[seed={seed}] Saved {output_path} | time={total_elapsed:.1f}s | rate={samples_per_sec:.1f} samples/sec"
    )
    return output_path


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parent
    policy_path = (project_root / args.policy_json).resolve()
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy JSON not found at {policy_path}")

    xml_path = (
        project_root.parent
        / "pupper_v3_description"
        / "description"
        / "mujoco_xml"
        / "pupper_v3_complete.mjx.position.xml"
    )
    output_dir = (project_root / args.output_dir).resolve()

    if args.num_workers <= 1:
        collect_dataset(
            policy_path,
            xml_path,
            output_dir,
            args.num_samples,
            args.seed,
            args.progress_steps,
            f"force_dataset_{args.num_samples}",
        )
        return

    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    num_workers = args.num_workers
    base_samples = args.num_samples // num_workers
    remainder = args.num_samples % num_workers

    worker_processes = []
    worker_paths = []

    for worker_idx in range(num_workers):
        samples = base_samples + (1 if worker_idx < remainder else 0)
        if samples == 0:
            continue
        worker_seed = args.seed + worker_idx
        dataset_name = f"force_dataset_worker{worker_idx}"
        worker_path = output_dir / f"{dataset_name}.npz"
        worker_paths.append(worker_path)

        proc = mp.Process(
            target=collect_dataset,
            args=(
                policy_path,
                xml_path,
                output_dir,
                samples,
                worker_seed,
                args.progress_steps,
                dataset_name,
            ),
        )
        proc.start()
        worker_processes.append(proc)

    for proc in worker_processes:
        proc.join()

    obs_list = []
    force_list = []
    for path in worker_paths:
        if not path.exists():
            continue
        data = np.load(path)
        obs_list.append(data["observations"])
        force_list.append(data["forces"])

    if not obs_list:
        raise RuntimeError("No worker produced data; check logs above.")

    observations_np = np.concatenate(obs_list, axis=0)[: args.num_samples]
    forces_np = np.concatenate(force_list, axis=0)[: args.num_samples]

    final_metadata = {
        "obs_dim": observations_np.shape[1],
        "num_samples": int(observations_np.shape[0]),
        "force_mean": forces_np.mean(axis=0).tolist(),
        "force_std": forces_np.std(axis=0).tolist(),
    }

    final_path = output_dir / f"force_dataset_{args.num_samples}.npz"
    np.savez_compressed(
        final_path,
        observations=observations_np,
        forces=forces_np,
        metadata=json.dumps(final_metadata),
    )
    print(f"Combined dataset written to {final_path}")


if __name__ == "__main__":
    main()



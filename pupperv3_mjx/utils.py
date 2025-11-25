from datetime import datetime
import matplotlib.pyplot as plt
import difflib
import re
import xml.etree.ElementTree as ET
from typing import List, Callable, Tuple, Optional, Sequence
import mediapy as media
import os
import wandb
import jax
from jax import numpy as jp
import numpy as np
import mujoco

from flax.training import orbax_utils
from orbax import checkpoint as ocp
from pathlib import Path


def circular_buffer_push_back(buffer: jax.Array, new_value: jax.Array) -> jax.Array:
    """
    Shift a circular buffer back by one step and set the last element to a new value.
    The newest element will be at buf[:, -1]

    Args:
        buffer (jax.Array): The circular buffer. Dimensions: (buffer_size, buffer_shape).
        new_value (jax.Array): The new value to set at the last index. Dimensions: (buffer_shape).
    Returns:
        jax.Array: The updated circular buffer.
    """
    buffer = jp.roll(buffer, shift=-1, axis=1)
    return buffer.at[:, -1].set(new_value)


def circular_buffer_push_front(buffer: jax.Array, new_value: jax.Array) -> jax.Array:
    """
    Shift a circular buffer forward by one step and set the first element to a new value.
    The newest element will be at buf[:, 0]

    Args:
        buffer (jax.Array): The circular buffer. Dimensions: (buffer_size, buffer_shape).
        new_value (jax.Array): The new value to set at the first index. Dimensions: (buffer_shape).
    Returns:
        jax.Array: The updated circular buffer.
    """
    buffer = jp.roll(buffer, shift=1, axis=1)
    return buffer.at[:, 0].set(new_value)


def sample_lagged_value(
    rng: jax.Array, buffer_newest_first: jax.Array, new_value: jax.Array, distribution: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """
    Sample a value from a circular buffer with a lagged distribution.
    Args:
        rng (jax.Array): The random number generator key.
        buffer_newest_first (jax.Array): The circular buffer with the newest element up front.
        new_value (jax.Array): The new value to set at the first index.
        distribution (jax.Array): The distribution to sample the lagged value from.
    Returns:
        Tuple[jax.Array, jax.Array]: The sampled value and the updated circular buffer.
    """
    buffer_newest_first = circular_buffer_push_front(buffer_newest_first, new_value)
    return jax.random.choice(rng, buffer_newest_first, axis=1, p=distribution), buffer_newest_first


def progress(
    num_steps: int,
    metrics: dict,
    times: list,
    x_data: list,
    y_data: list,
    ydataerr: list,
    num_timesteps: int,
    min_y: float,
    max_y: float,
):
    """
    Update and display a progress plot with error bars.

    Args:
    num_steps (int): The current number of steps in the environment.
    metrics (dict): A dictionary containing evaluation metrics.
    times (list): A list to append the current time.
    x_data (list): A list to append the current number of steps.
    y_data (list): A list to append the current episode reward.
    ydataerr (list): A list to append the standard deviation of the episode reward.
    num_timesteps (int): The total number of timesteps for the x-axis limit.
    min_y (float): The minimum y-axis value.
    max_y (float): The maximum y-axis value.
    """
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    ydataerr.append(metrics["eval/episode_reward_std"])

    plt.xlim([0, num_timesteps * 1.25])
    plt.ylim([min_y, max_y])

    plt.xlabel("# environment steps")
    plt.ylabel("reward per episode")
    plt.title(f"y={y_data[-1]:.3f}")

    plt.errorbar(x_data, y_data, yerr=ydataerr)
    plt.show()

    wandb.log(metrics, step=num_steps)


def fuzzy_search(obj, search_str: str, cutoff: float = 0.6):
    """
    Perform a fuzzy search on the properties of an object.

    Args:
    obj: The object to search through.
    search_str (str): The string to match properties against.
    cutoff (float): The cutoff for matching ratio (0.0 to 1.0), higher means more strict matching.

    Returns:
    List[Tuple[str, float]]: A list of tuples containing (property_name, match_ratio) that match
    the search string.
    """
    results = []

    # Get all properties of the object
    properties = dir(obj)

    # Search for fuzzy matches
    for prop in properties:
        ratio = difflib.SequenceMatcher(None, search_str, prop).ratio()
        if ratio >= cutoff:
            results.append((prop, ratio))

    # Sort results by match ratio in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    return results


def set_mjx_custom_options(tree: ET.ElementTree, max_contact_points: int, max_geom_pairs: int):
    """
    Set custom options for a MuJoCo XML model.

    Args:
    tree (ET.ElementTree): The XML tree of the MuJoCo model.
    max_contact_points (int): The maximum number of contact points.
    max_geom_pairs (int): The maximum number of geometry pairs.

    Returns:
    ET.ElementTree: The updated XML tree.
    """
    root = tree.getroot()
    custom = root.find("custom")
    if custom is not None:
        for numeric in custom.findall("numeric"):
            name = numeric.get("name")
            if name == "max_contact_points":
                numeric.set("data", str(max_contact_points))
            elif name == "max_geom_pairs":
                numeric.set("data", str(max_geom_pairs))

        return tree
    return None


def set_robot_starting_position(
    tree: ET.ElementTree, starting_pos: List, starting_quat: List = None
):
    """
    Change the starting position of the robot in the XML MuJoCo model file.

    Args:
    tree (ET.ElementTree): The XML tree of the MuJoCo model.
    starting_pos (List[float]): The starting position [x, y, z].
    starting_quat (List[float], optional): The starting quaternion [x, y, z, w].

    Returns:
    ET.ElementTree: The updated XML tree.
    """

    body = tree.find(".//worldbody/body[@name='base_link']")
    body.set("pos", f"{starting_pos[0]} {starting_pos[1]} {starting_pos[2]}")
    if starting_quat is not None:
        body.set(
            "quat", f"{starting_quat[0]} {starting_quat[1]} {starting_quat[2]} {starting_quat[3]}"
        )

    home_position = tree.find(".//keyframe/key[@name='home']")
    qpos_scalar = list(map(float, re.split(r"\s+", home_position.get("qpos").strip())))
    qpos_scalar[:3] = starting_pos
    if starting_quat is not None:
        qpos_scalar[3:7] = starting_quat
    updated_qpos = " ".join(map(str, qpos_scalar))
    home_position.set("qpos", updated_qpos)
    return tree


def save_checkpoint(current_step, make_policy, params, checkpoint_path: Path):
    # save checkpoints
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    path = Path(checkpoint_path) / Path(f"{current_step}")
    orbax_checkpointer.save(path.resolve(), params, force=True, save_args=save_args)
    wandb.log_model(path=path.as_posix(), name=f"checkpoint_{wandb.run.name}_{current_step}")


def visualize_policy(
    current_step,
    make_policy,
    params,
    eval_env,
    jit_step: Callable,
    jit_reset: Callable,
    output_folder: str,
    vx: float = 0.5,
    vy: float = 0.4,
    wz: float = 1.5,
):
    """
    Visualize a policy by creating a video of the robot's behavior.

    Args:
    current_step (int): The current training step.
    make_policy (Callable): A function to create the policy.
    params (Tuple): The parameters for the policy.
    eval_env: The evaluation environment.
    jit_step (Callable): A JIT-compiled function to perform a step in the environment.
    jit_reset (Callable): A JIT-compiled function to reset the environment.
    output_folder (str): The folder to save the output video.
    vx (float): The forward/backward velocity.
    vy (float): The left/right velocity.
    wz (float): The rotational velocity.
    """

    inference_fn = make_policy((params[0], params[1].policy))
    jit_inference_fn = jax.jit(inference_fn)

    # Make robot go forward, back, left, right
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

    # initialize the state
    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    state.info["command"] = command_seq[0]
    rollout = [state.pipeline_state]

    # grab a trajectory
    n_steps = 560
    render_every = 2
    ctrls = []

    for i in range(n_steps):
        act_rng, rng = jax.random.split(rng)

        # Change command every 80 steps
        state.info["command"] = command_seq[int(i / 80)]

        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state)
        ctrls.append(ctrl)

    filename = os.path.join(output_folder, f"step_{current_step}_policy.mp4")
    fps = int(1.0 / eval_env.dt / render_every)
    media.write_video(
        filename,
        eval_env.render(rollout[::render_every], camera="tracking_cam"),
        fps=fps,
    )
    wandb.log(
        {
            "eval/video/command/vx": vx,
            "eval/video/command/vy": vy,
            "eval/video/command/wz": wz,
            "eval/video": wandb.Video(filename, format="mp4"),
        },
        step=current_step,
    )


def activation_fn_map(activation_name: str):
    """
    Map an activation function name to its corresponding JAX function.

    Args:
    activation_name (str): The name of the activation function (e.g., 'relu', 'sigmoid').

    Returns:
    Callable: The corresponding JAX activation function.
    """
    activation_name = activation_name.lower()
    return {
        "relu": jax.nn.relu,
        "sigmoid": jax.nn.sigmoid,
        "elu": jax.nn.elu,
        "tanh": jp.tanh,
        "softmax": jax.nn.softmax,
    }[activation_name]


def visualize_force_arrow(
    renderer: mujoco.Renderer,
    origin: np.ndarray,
    force: np.ndarray,
    vis_scale: float = 0.1,
    rgba: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
) -> None:
    """Adds a force arrow to the MuJoCo scene.

    Args:
        renderer: The MuJoCo Renderer instance (scene is modified in-place).
        origin: 3D world position where the arrow starts.
        force: 3D force vector (direction and magnitude).
        vis_scale: Scaling factor to convert force magnitude to arrow length.
        rgba: RGBA color of the arrow.
    """
    p1 = np.asarray(origin, dtype=np.float64)
    p2 = p1 + np.asarray(force, dtype=np.float64) * vis_scale

    i = renderer.scene.ngeom
    if i >= renderer.scene.maxgeom:
        return  # scene full

    geom = renderer.scene.geoms[i]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.01, p1, p2)
    renderer.scene.ngeom += 1


def render_leash_rollout(
    model_path: str,
    rollout: List,
    leash_positions: List[np.ndarray],
    width: int = 640,
    height: int = 480,
    camera: str = "tracking_cam",
    leash_attachment_point: Optional[np.ndarray] = None,
    force_vis_scale: float = 0.1,
    render_every: int = 1,
) -> Sequence[np.ndarray]:
    """Renders a rollout with leash target ball and force arrow visualization.

    Args:
        model_path: Path to the MJCF model file.
        rollout: List of pipeline states from the simulation.
        leash_positions: List of leash target positions (one per frame in rollout).
        width: Rendered frame width in pixels.
        height: Rendered frame height in pixels.
        camera: Camera name defined in the MJCF.
        leash_attachment_point: Local offset on the torso where the leash attaches.
        force_vis_scale: Scaling factor for force arrow length.
        render_every: Render every Nth frame.

    Returns:
        List of rendered frames as numpy arrays.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # Find the leash site ID in the model
    leash_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "leash_target_site")
    if leash_site_id < 0:
        print("Warning: leash_target_site not found in model, target sphere won't be visible")

    renderer = None
    gl_context = None
    try:
        gl_context = mujoco.GLContext(max_width=width, max_height=height)
        gl_context.make_current()
        renderer = mujoco.Renderer(model, width=width, height=height)
    except Exception as err:
        if renderer is not None:
            renderer.close()
        if gl_context is not None:
            gl_context.free()
        raise RuntimeError(
            "Failed to initialize MuJoCo OpenGL renderer. "
            "Install EGL-compatible drivers or run within a graphical environment."
        ) from err

    option = mujoco.MjvOption()
    # Enable site group 3 where the leash marker is placed
    option.sitegroup[3] = True

    frames = []
    try:
        # Subsample both rollout and leash_positions together
        subsampled_indices = range(0, len(rollout), render_every)
        for idx in subsampled_indices:
            ps = rollout[idx]
            leash_pos = leash_positions[idx]

            data = mujoco.MjData(model)
            mujoco.mj_resetData(model, data)
            data.qpos[:] = np.asarray(ps.q)
            data.qvel[:] = np.asarray(ps.qd)
            data.xfrc_applied[:] = np.asarray(ps.xfrc_applied)
            mujoco.mj_forward(model, data)

            # Update the leash target site position in MuJoCo data
            if leash_site_id >= 0:
                data.site_xpos[leash_site_id] = leash_pos

            renderer.update_scene(data, camera=camera, scene_option=option)

            # Draw leash force arrow if attachment point provided
            if leash_attachment_point is not None:
                # xfrc_applied is (nbody, 6): [torque (3), force (3)]
                # torso is body index 1 (world is 0)
                torso_idx = 1
                force_world = np.asarray(ps.xfrc_applied[torso_idx, 3:6])
                if np.linalg.norm(force_world) > 1e-3:
                    # Compute world-frame attachment point
                    torso_pos = data.xpos[torso_idx]
                    torso_rot = data.xmat[torso_idx].reshape(3, 3)
                    attach_world = torso_pos + torso_rot @ np.asarray(leash_attachment_point)
                    visualize_force_arrow(
                        renderer,
                        origin=attach_world,
                        force=force_world,
                        vis_scale=force_vis_scale,
                        rgba=(1.0, 0.3, 0.0, 1.0),  # orange arrow
                    )

            frame = renderer.render()
            frames.append(np.asarray(frame))
    finally:
        if renderer is not None:
            renderer.close()
        if gl_context is not None:
            gl_context.free()

    return frames


def visualize_policy_with_leash(
    current_step,
    make_policy,
    params,
    eval_env,
    jit_step: Callable,
    jit_reset: Callable,
    output_folder: str,
    model_path: str,
    leash_attachment_point: np.ndarray,
    vx: float = 0.5,
    vy: float = 0.4,
    wz: float = 1.5,
    force_vis_scale: float = 0.1,
):
    """
    Visualize a policy with leash target ball and force arrow.

    Args:
        current_step: The current training step.
        make_policy: A function to create the policy.
        params: The parameters for the policy.
        eval_env: The evaluation environment.
        jit_step: A JIT-compiled function to perform a step in the environment.
        jit_reset: A JIT-compiled function to reset the environment.
        output_folder: The folder to save the output video.
        model_path: Path to the MJCF model file.
        leash_attachment_point: Local offset on the torso where the leash attaches.
        vx: The forward/backward velocity.
        vy: The left/right velocity.
        wz: The rotational velocity.
        force_vis_scale: Scaling factor for force arrow length.
    """
    inference_fn = make_policy((params[0], params[1].policy))
    jit_inference_fn = jax.jit(inference_fn)

    # Make robot go forward, back, left, right
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

    # initialize the state
    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    state.info["command"] = command_seq[0]
    rollout = [state.pipeline_state]
    leash_positions = [np.asarray(state.info["leash_target_pos"])]

    # grab a trajectory
    n_steps = 560
    render_every = 2

    for i in range(n_steps):
        act_rng, rng = jax.random.split(rng)

        # Change command every 80 steps
        state.info["command"] = command_seq[int(i / 80)]

        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state)
        leash_positions.append(np.asarray(state.info["leash_target_pos"]))

    # Render with leash visualization
    frames = render_leash_rollout(
        model_path=model_path,
        rollout=rollout,
        leash_positions=leash_positions,
        camera="tracking_cam",
        leash_attachment_point=leash_attachment_point,
        force_vis_scale=force_vis_scale,
        render_every=render_every,
    )

    filename = os.path.join(output_folder, f"step_{current_step}_policy_leash.mp4")
    fps = int(1.0 / eval_env.dt / render_every)
    media.write_video(filename, frames, fps=fps)

    wandb.log(
        {
            "eval/video/command/vx": vx,
            "eval/video/command/vy": vy,
            "eval/video/command/wz": wz,
            "eval/video_leash": wandb.Video(filename, format="mp4"),
        },
        step=current_step,
    )


def download_checkpoint(
    project_name,
    entity_name,
    run_number: int,
    save_path: Path = Path("checkpoint"),
):
    """
    Downloads the latest model from a W&B project.

    :param project_name: The name of the W&B project.
    :param entity_name: The W&B entity (username or team).
    :param model_dir: The directory where the model will be downloaded.
    :param model_name: The name to copy the model as.
    :return: None
    """

    # Initialize the API
    api = wandb.Api()

    # Fetch the latest run
    runs = api.runs(f"{entity_name}/{project_name}")

    # Check if there are any runs
    if not runs:
        print("No runs found in the project.")
        return

    # find the run whose names ends in -run_number
    runs = [run for run in runs if run.name.endswith(f"-{run_number}")]
    if not runs:
        print(f"No runs found with the number {run_number}.")
        return
    run = runs[0]
    print("Using run: ", run.name)

    # get artifacts that start with "checkpoint"
    artifacts = [art for art in run.logged_artifacts() if "checkpoint" in art.name]

    # sort by the number at the end which has a _ before it and a :blah after it
    artifacts = sorted(
        artifacts,
        key=lambda art: int(art.name.split("_")[-1].split(":")[0]),
        reverse=True,
    )
    latest_checkpoint = artifacts[0]

    print("Downloading the latest checkpoint: ", latest_checkpoint.name, " to ", save_path)
    latest_checkpoint.download(save_path)

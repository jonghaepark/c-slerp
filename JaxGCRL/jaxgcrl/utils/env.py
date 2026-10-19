import argparse
import inspect
import json
import logging
import math
import os
from collections import namedtuple
from datetime import datetime
from typing import List

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import wandb_osh
from brax.io import html
from matplotlib import pyplot as plt
from wandb_osh.hooks import TriggerWandbSyncHook

import wandb
from jaxgcrl.envs.ant import Ant
from jaxgcrl.envs.ant_ball import AntBall
from jaxgcrl.envs.ant_ball_maze import AntBallMaze
from jaxgcrl.envs.ant_maze import AntMaze
from jaxgcrl.envs.ant_push import AntPush
from jaxgcrl.envs.half_cheetah import Halfcheetah
from jaxgcrl.envs.humanoid import Humanoid
from jaxgcrl.envs.humanoid_maze import HumanoidMaze
from jaxgcrl.envs.manipulation.arm_binpick_easy import ArmBinpickEasy
from jaxgcrl.envs.manipulation.arm_binpick_hard import ArmBinpickHard
from jaxgcrl.envs.manipulation.arm_binpick_opposite import ArmBinpickOpposite
from jaxgcrl.envs.manipulation.arm_grasp import ArmGrasp
from jaxgcrl.envs.manipulation.arm_push_easy import ArmPushEasy
from jaxgcrl.envs.manipulation.arm_push_hard import ArmPushHard
from jaxgcrl.envs.manipulation.arm_push_opposite import ArmPushOpposite
from jaxgcrl.envs.manipulation.arm_reach import ArmReach
from jaxgcrl.envs.pusher import Pusher, PusherReacher
from jaxgcrl.envs.pusher2 import Pusher2
from jaxgcrl.envs.reacher import Reacher
from jaxgcrl.envs.simple_maze import SimpleMaze

legal_envs = (
    "ant",
    "ant_random_start",
    "ant_ball",
    "ant_push",
    "humanoid",
    "humanoid_extreme",
    "humanoid_extreme_eval",
    "humanoid_extreme_unidirect",
    "humanoid_extreme_unidirect_eval",
    "reacher",
    "cheetah",
    "pusher_easy",
    "pusher_hard",
    "pusher_reacher",
    "pusher2",
    "arm_reach",
    "arm_grasp",
    "arm_push_easy",
    "arm_push_hard",
    "arm_push_opposite",
    "arm_binpick_easy",
    "arm_binpick_hard",
    "arm_binpick_opposite",
    "ant_ball_maze",
    "ant_u_maze",
    "ant_u_maze_eval",
    "ant_u4_maze",
    "ant_u4_maze_eval",
    "ant_u4_maze_single",
    "ant_u5_maze",
    "ant_u5_maze_eval",
    "ant_big_maze",
    "ant_big_maze_eval",
    "ant_big_maze_single",
    "ant_hardest_maze",
    "ant_hardest_maze_eval",
    "ant_spiral_maze_easy",
    "ant_spiral_maze_easy_eval",
    "ant_spiral_maze_hard",
    "ant_spiral_maze_hard_eval",
    "ant_extreme_maze",
    "ant_extreme_maze_eval",
    "ant_extreme_maze_v2",
    "ant_extreme_maze_v2_eval",
    "ant_dfs_maze",
    "ant_dfs_maze_eval",
    "humanoid_u_maze",
    "humanoid_u_maze_eval",
    "humanoid_big_maze",
    "humanoid_big_maze_eval",
    "humanoid_hardest_maze",
    "simple_u_maze",
    "simple_big_maze",
    "simple_hardest_maze",
)


def create_env(env_name: str, backend: str = None, **kwargs) -> object:
    """
    This function creates and returns an appropriate environment object based on the specified environment name and
    backend.

    Args:
        env_name (str): Name of the environment.
        backend (str): Backend to be used for the environment.

    Returns:
        object: The instantiated environment object.

    Raises:
        ValueError: If the specified environment name is unknown.
    """
    if env_name == "reacher":
        env = Reacher(backend=backend or "generalized")
    elif env_name == "ant":
        env = Ant(backend=backend or "spring")
    elif env_name == "ant_random_start":
        env = Ant(backend=backend or "spring", randomize_start=True)
    elif env_name == "ant_ball":
        env = AntBall(backend=backend or "spring")
    elif env_name == "ant_push":
        # This is stable only in mjx backend
        assert backend == "mjx" or backend is None
        env = AntPush(backend=backend or "mjx")
    elif "maze" in env_name:
        if "ant_ball" in env_name:
            env = AntBallMaze(backend=backend or "spring", maze_layout_name=env_name[9:])
        elif "ant" in env_name:
            # Possible env_name = {'ant_u_maze', 'ant_big_maze', 'ant_hardest_maze', ...}
            env = AntMaze(backend=backend or "spring", maze_layout_name=env_name[4:])#, terminate_when_unhealthy=False)
        elif "humanoid" in env_name:
            # Possible env_name = {'humanoid_u_maze', 'humanoid_big_maze', 'humanoid_hardest_maze',
            #                      'humanoid_u_maze_eval', 'humanoid_big_maze_eval'}
            env = HumanoidMaze(backend=backend or "spring", maze_layout_name=env_name[9:])
        else:
            # Possible env_name = {'simple_u_maze', 'simple_big_maze', 'simple_hardest_maze'}
            env = SimpleMaze(backend=backend or "spring", maze_layout_name=env_name[7:])
    elif env_name == "cheetah":
        env = Halfcheetah()
    elif env_name == "pusher_easy":
        env = Pusher(backend=backend or "generalized", kind="easy")
    elif env_name == "pusher_hard":
        env = Pusher(backend=backend or "generalized", kind="hard")
    elif env_name == "pusher_reacher":
        env = PusherReacher(backend=backend or "generalized")
    elif env_name == "pusher2":
        env = Pusher2(backend=backend or "generalized")
    elif env_name == "humanoid":
        env = Humanoid(backend=backend or "spring")
    elif env_name == "humanoid_extreme":
        env = Humanoid(
            backend=backend or "spring",
            asset_file="humanoid_extreme.xml",
            min_goal_dist=1.0,
            max_goal_dist=50.0,
            uniform_goal_sampling=True,
        )
    elif env_name == "humanoid_extreme_eval":
        env = Humanoid(
            backend=backend or "spring",
            asset_file="humanoid_extreme.xml",
            min_goal_dist=10.0,
            max_goal_dist=50.0,
            uniform_goal_sampling=True,
        )
    elif env_name == "humanoid_extreme_unidirect":
        env = Humanoid(
            backend=backend or "spring",
            asset_file="humanoid_extreme.xml",
            min_goal_dist=1.0,
            max_goal_dist=20.0,
            target_sampling="unidirectional",
            fixed_goal_angle_deg=0.0,
        )
    elif env_name == "humanoid_extreme_unidirect_eval":
        env = Humanoid(
            backend=backend or "spring",
            asset_file="humanoid_extreme.xml",
            fixed_goal_angle_deg=0.0,
            possible_goal_distances=tuple(range(2, 21, 2)),
        )
    elif env_name == "arm_reach":
        env = ArmReach(backend=backend or "mjx")
    elif env_name == "arm_grasp":
        env = ArmGrasp(backend=backend or "mjx")
    elif env_name == "arm_push_easy":
        env = ArmPushEasy(backend=backend or "mjx")
    elif env_name == "arm_push_hard":
        env = ArmPushHard(backend=backend or "mjx")
    elif env_name == "arm_push_opposite":
        env = ArmPushOpposite(backend=backend or "mjx")
    elif env_name == "arm_binpick_easy":
        env = ArmBinpickEasy(backend=backend or "mjx")
    elif env_name == "arm_binpick_hard":
        env = ArmBinpickHard(backend=backend or "mjx")
    elif env_name == "arm_binpick_opposite":
        env = ArmBinpickOpposite(backend=backend or "mjx")
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    return env


def get_env_config(args: argparse.Namespace):
    """
    Generate and validate environment configuration based on input arguments.

    This function takes an argparse.Namespace object, validates the specified environment name
    against a list of legal environments, and returns a configuration named tuple constructed
    from the input arguments.

    Parameters
    ----------
    args : dataclass
        The input arguments containing the environment name and other configuration settings.

    Returns
    -------
    Config
        A named tuple containing the configuration derived from the input arguments.

    Raises
    ------
    ValueError
        If the specified environment name is not in the list of legal environments or does not
        contain the word 'maze'.
    """
    if args.env_name not in legal_envs:
        raise ValueError(f"Unknown environment: {args.env_name}")

    # TODO: round num_envs to nearest valid value instead of throwing error
    if ((args.episode_length - 1) * args.num_envs) % args.batch_size != 0:
        raise ValueError("(episode_length - 1) * num_envs must be divisible by batch_size")

    args_dict = vars(args)
    Config = namedtuple("Config", [*args_dict.keys()])
    config = Config(*args_dict.values())

    return config


class MetricsRecorder:
    """
    Initialize the MetricsRecorder with the specified number of timesteps
    and the metrics to be collected.

    Parameters:
    total_env_steps (int): The maximum number of timesteps for recording metrics.
    metrics_to_collect (List[str]): List of metric names that are to be collected.
    exp_dir (str): Directory to save renders to.
    exp_name (str): Experiment name for naming rendered trajectory visualizations.
    """

    def __init__(
        self,
        total_env_steps: int,
        metrics_to_collect: List[str],
        exp_dir,
        exp_name,
        mode,
        episode_length: int = 1001,
    ):
        self.x_data = []
        self.y_data = {}
        self.y_data_err = {}
        self.times = [datetime.now()]
        self.metrics_to_collect = metrics_to_collect
        self.exp_dir = exp_dir
        self.exp_name = exp_name
        self.mode = mode
        self.episode_length = episode_length

        self.max_x, self.min_x = total_env_steps * 1.1, 0

        if mode == "offline":
            wandb_osh.set_log_level("ERROR")
            self.trigger_sync = TriggerWandbSyncHook()
        else:
            self.trigger_sync = None

    def record(self, num_steps, metrics):
        self.times.append(datetime.now())
        self.x_data.append(int(num_steps))

        for key, value in metrics.items():
            if key not in self.y_data:
                self.y_data[key] = []
                self.y_data_err[key] = []

            self.y_data[key].append(value)
            self.y_data_err[key].append(metrics.get(f"{key}_std", 0))

    def log_wandb(self):
        if self.mode == "disabled":
            return

        data_to_log = {}
        for key, value in self.y_data.items():
            data_to_log[key] = value[-1]
        data_to_log["step"] = self.x_data[-1]
        wandb.log(data_to_log, step=self.x_data[-1])

        if self.mode == "offline" and self.trigger_sync is not None:
            self.trigger_sync()

    def plot_progress(self):
        num_plots = len(self.y_data)
        # Calculate number of rows needed for 2 columns
        num_rows = (num_plots + 1) // 2

        fig, axs = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))

        for idx, (key, y_values) in enumerate(self.y_data.items()):
            row = idx // 2
            col = idx % 2

            axs[row, col].set_xlim(self.min_x, self.max_x)
            axs[row, col].set_xlabel("# environment steps")
            axs[row, col].set_ylabel(key)
            axs[row, col].errorbar(self.x_data, y_values, yerr=self.y_data_err[key])
            axs[row, col].set_title(f"{key}: {y_values[-1]:.3f}")

        # Hide any empty subplots
        for idx in range(num_plots, num_rows * 2):
            row = idx // 2
            col = idx % 2
            axs[row, col].axis("off")
        plt.tight_layout()
        plt.show()

    def print_progress(self):
        for idx, (key, y_values) in enumerate(self.y_data.items()):
            logging.info(
                f"step: {self.x_data[-1]}, {key}: {y_values[-1]:.3f} +/- {self.y_data_err[key][-1]:.3f}"
            )

    def print_times(self):
        logging.info(f"time to jit: {self.times[1] - self.times[0]}")
        logging.info(f"time to train: {self.times[-1] - self.times[1]}")

    def progress(self, num_steps, metrics, make_policy, params, env, do_render=True):
        for key in self.metrics_to_collect:
            self.ensure_metric(metrics, key)

        if do_render:
            render(make_policy, params, env, self.exp_dir, self.exp_name, num_steps,
                   wandb_step=int(num_steps),
                   episode_length=self.episode_length)

        self.record(
            num_steps,
            {key: value for key, value in metrics.items() if key in self.metrics_to_collect},
        )
        self.log_wandb()
        self.print_progress()

    @staticmethod
    def ensure_metric(metrics, key):
        if key not in metrics:
            metrics[key] = 0
        else:
            if math.isnan(metrics[key]):
                raise Exception(f"Metric: {key} is NaN in metrics: {metrics}")


def _topdown_camera_patch(html_str, env):
    """Inject top-down camera setup into Brax HTML viewer.

    Computes the environment center from the env's goals and starts,
    then repositions the Three.js camera directly above looking down.
    Disables fog and extends clipping planes so the full maze is visible.
    """
    start_points = (
        np.asarray(env.possible_starts)
        if hasattr(env, "possible_starts")
        else np.asarray(env.start_modes)
    )
    all_pts = np.concatenate([
        np.asarray(env.possible_goals),
        start_points,
    ], axis=0)
    cx = float(np.mean(all_pts[:, 0]))
    cz = float(np.mean(all_pts[:, 1]))
    span = float(max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1])))
    cam_height = span * 1.8 + 15

    cam_js = "\n".join([
        f"viewer.camera.follow = false;",
        f"      viewer.camera.far = 500;",
        f"      viewer.camera.position.set({cx}, {cz}, {cam_height});",
        f"      viewer.controls.target.set({cx}, {cz}, 0);",
        f"      viewer.controls.enablePan = true;",
        f"      viewer.camera.updateProjectionMatrix();",
        f"      viewer.controls.update();",
        f"      viewer.scene.fog = null;",
        f"      viewer.setDirty();",
    ])

    old = "var viewer = new Viewer(domElement, system);"
    new = f"var viewer = new Viewer(domElement, system);\n      {cam_js}"
    return html_str.replace(old, new)


def _pca_project_blocks(blocks, fit_blocks=None):
    """Projects latent blocks using one shared standardized PCA fit."""
    names = [name for name, _ in blocks]
    arrays = [np.asarray(arr, dtype=np.float32) for _, arr in blocks]
    lengths = [len(arr) for arr in arrays]
    if not arrays or sum(lengths) == 0:
        return {name: np.zeros((0, 2), dtype=np.float32) for name in names}

    fit_arrays = arrays if fit_blocks is None else [
        np.asarray(arr, dtype=np.float32) for _, arr in fit_blocks
    ]
    fit_combined = np.concatenate(fit_arrays, axis=0)
    mean = fit_combined.mean(axis=0, keepdims=True)
    std = fit_combined.std(axis=0, keepdims=True)
    safe_std = np.where(std > 1e-6, std, 1.0)
    fit_scaled = (fit_combined - mean) / safe_std
    combined_scaled = (np.concatenate(arrays, axis=0) - mean) / safe_std

    if fit_scaled.shape[1] == 1:
        coords = np.concatenate(
            [combined_scaled, np.zeros_like(combined_scaled)], axis=1
        )
    else:
        _, _, vh = np.linalg.svd(fit_scaled, full_matrices=False)
        coords = combined_scaled @ vh[: min(2, vh.shape[0])].T
        if coords.shape[1] == 1:
            coords = np.concatenate([coords, np.zeros_like(coords)], axis=1)

    projected = {}
    cursor = 0
    for name, length in zip(names, lengths):
        projected[name] = coords[cursor: cursor + length]
        cursor += length
    return projected


def _latent_pca_blocks(segment):
    """Builds projection and deduplicated fit blocks for one rollout."""
    psi_s_np = segment["psi_s"]
    psi_g_np = segment["psi_g"]
    waypoint_np = segment["waypoint"]
    candidates_np = segment["candidates"]
    candidate_valid = segment["candidate_valid"]
    wp_update = segment["wp_update"]

    project_blocks = [
        ("psi_s", psi_s_np),
        ("psi_g", psi_g_np),
        ("waypoint", waypoint_np),
        (
            "candidates",
            candidates_np.reshape(-1, waypoint_np.shape[-1]),
        ),
    ]

    # State latents use every frame; the fixed goal appears once; planner
    # outputs only enter at actual high-level update steps to avoid
    # control-horizon duplicates.
    fit_blocks = [("psi_s", psi_s_np)]
    if len(psi_g_np):
        fit_blocks.append(("psi_g", psi_g_np[:1]))
    if np.any(wp_update):
        fit_blocks.append(("waypoint", waypoint_np[wp_update]))
        if candidates_np.shape[1] > 0:
            update_candidates = candidates_np[wp_update]
            update_valid = candidate_valid[wp_update]
            valid_candidates = update_candidates[update_valid]
            if len(valid_candidates):
                fit_blocks.append(("candidates", valid_candidates))
    return project_blocks, fit_blocks


def _project_latent_segment(segment):
    """Fits a rollout-local PCA and serializes the projected panel data."""
    project_blocks, fit_blocks = _latent_pca_blocks(segment)
    projected = _pca_project_blocks(project_blocks, fit_blocks=fit_blocks)
    seg_len = segment["length"]
    num_candidates = segment["candidates"].shape[1]
    candidate_coords = projected["candidates"].reshape(
        seg_len,
        num_candidates,
        2,
    )
    return {
        "traj_idx": segment["traj_idx"],
        "start": segment["start"],
        "end": segment["end"],
        "length": int(seg_len),
        "psi_s": projected["psi_s"].tolist(),
        "psi_g": projected["psi_g"].tolist(),
        "waypoint": projected["waypoint"].tolist(),
        "candidates": candidate_coords.tolist(),
        "candidate_valid": segment["candidate_valid"].tolist(),
        "planning_k": segment["planning_k"].tolist(),
        "planning_i": segment["planning_i"].tolist(),
        "k_probs": segment["k_probs"].tolist(),
        "wp_update": segment["wp_update"].tolist(),
    }


def _visualization_rollout_count(env):
    specs = getattr(env, "visualization_start_goal_indices", None)
    if specs is not None:
        return len(specs)
    configured_rollouts = getattr(env, "visualization_rollouts", None)
    if configured_rollouts is not None:
        return int(configured_rollouts)
    if hasattr(env, "possible_goals"):
        return len(env.possible_goals)
    if str(getattr(env, "env_name", "")).startswith("arm_"):
        return 4
    return 1


def _apply_visualization_start_goal(env, state, key, rollout_idx):
    specs = getattr(env, "visualization_start_goal_indices", None)
    target = None

    def _get_obs_after_pipeline_edit(pipeline_state):
        signature = inspect.signature(env._get_obs)
        params = list(signature.parameters)
        if len(params) == 1:
            return env._get_obs(pipeline_state)
        if len(params) == 2 and params[1] == "action":
            return env._get_obs(pipeline_state, jnp.zeros(env.sys.act_size()))
        if len(params) == 3 and "goal" in params and "timestep" in params:
            return env._get_obs(
                pipeline_state,
                state.info["goal"],
                state.info["timestep"],
            )
        raise TypeError(f"Unsupported _get_obs signature for {type(env).__name__}: {signature}")

    if specs is not None:
        start_idx, goal_idx = specs[rollout_idx]
        if hasattr(env, "sample_visualization_start"):
            key, start_key = jax.random.split(key)
            start = env.sample_visualization_start(start_idx, start_key)
        else:
            start = env.start_modes[start_idx]

        if hasattr(env, "sample_visualization_goal"):
            key, goal_key = jax.random.split(key)
            target = env.sample_visualization_goal(goal_idx, goal_key)
        else:
            target = env.possible_goals[goal_idx]

        q = state.pipeline_state.q.at[:2].set(start)
        q = q.at[-2:].set(target)
        new_ps = env.pipeline_init(q, state.pipeline_state.qd)
        obs = _get_obs_after_pipeline_edit(new_ps)
        return state.replace(pipeline_state=new_ps, obs=obs), key, target

    if hasattr(env, "possible_goals"):
        if hasattr(env, "sample_visualization_goal"):
            key, goal_key = jax.random.split(key)
            target = env.sample_visualization_goal(rollout_idx, goal_key)
        else:
            target = env.possible_goals[rollout_idx]
        q = state.pipeline_state.q.at[-2:].set(target)
        new_ps = env.pipeline_init(q, state.pipeline_state.qd)
        obs = _get_obs_after_pipeline_edit(new_ps)
        state = state.replace(pipeline_state=new_ps, obs=obs)

    return state, key, target


def _inject_latent_sync_panel(html_str, sync_data):
    layout_html = """
<div id="sync-layout">
  <div id="viewer-pane"><div id="brax-viewer"></div></div>
  <div id="latent-pane" class="planner-pane">
    <div class="planner-title">Recursive CRL Latent Planner</div>
    <div id="latent-meta" class="planner-meta"></div>
    <canvas id="latent-canvas" class="planner-canvas" width="720" height="720"></canvas>
    <div class="planner-legend">
      <span class="legend-item" style="color:#111111"><span class="legend-dot" style="background:#111111"></span><b>&psi;(s)</b></span>
      <span class="legend-item" style="color:#CB297B"><span class="legend-dot" style="background:#CB297B"></span><b>&psi;(g)</b></span>
      <span class="legend-item" style="color:#5a82ff"><span class="legend-dot" style="background:#5a82ff"></span><b>selected waypoint</b></span>
    </div>
  </div>
  <div id="k-prob-pane" class="planner-pane">
    <div class="planner-title">Analytic K Distribution</div>
    <div id="k-prob-meta" class="planner-meta"></div>
    <canvas id="k-prob-canvas" class="planner-canvas" width="720" height="720"></canvas>
    <div class="planner-legend">
      <span class="legend-item" style="color:#5a82ff"><span class="legend-swatch" style="background:#5a82ff"></span><b>selected K*</b></span>
      <span class="legend-item" style="color:#777d84"><span class="legend-swatch" style="background:#a6abb1"></span><b>not selected</b></span>
    </div>
  </div>
</div>
"""

    style_html = """
<style>
  html, body { margin: 0; height: 100%; background: #101418; color: #e8eef5; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  #sync-layout { display: flex; min-height: 100vh; width: 100vw; overflow: hidden; }
  #viewer-pane { position: relative; flex: 1 1 auto; min-width: 0; background: #101418; }
  #brax-viewer { width: 100%; height: 100vh; }
  .planner-pane { flex: 0 0 min(600px, 30vw); box-sizing: border-box; padding: 18px; background: #f7f5f0; color: #1d252c; border-left: 1px solid rgba(0,0,0,0.12); }
  .planner-title { font-size: 22px; font-weight: 700; margin-bottom: 8px; }
  .planner-meta { font-size: 16px; line-height: 1.55; margin-bottom: 12px; min-height: 82px; }
  .planner-canvas { display: block; box-sizing: border-box; width: 100%; aspect-ratio: 1 / 1; background: #fffdf9; border: 1px solid rgba(0,0,0,0.10); }
  .planner-legend { display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 16px; margin-top: 12px; font-size: 14px; text-align: center; }
  .legend-item { display: inline-flex; align-items: center; gap: 6px; font-weight: 700; }
  .legend-dot { display: inline-block; width: 11px; height: 11px; border-radius: 50%; box-shadow: 0 0 0 1px rgba(0,0,0,0.12); }
  .legend-swatch { display: inline-block; width: 14px; height: 11px; box-shadow: 0 0 0 1px rgba(0,0,0,0.12); }
  .dg.ac { z-index: 10 !important; }
</style>
"""

    sync_js = r"""
const SYNC_DATA = __SYNC_DATA__;
const canvas = document.getElementById('latent-canvas');
const ctx = canvas.getContext('2d');
const meta = document.getElementById('latent-meta');
const kCanvas = document.getElementById('k-prob-canvas');
const kCtx = kCanvas.getContext('2d');
const kMeta = document.getElementById('k-prob-meta');
const cssWidth = canvas.width;
const cssHeight = canvas.height;
const kCssWidth = kCanvas.width;
const kCssHeight = kCanvas.height;
const dpr = window.devicePixelRatio || 1;
canvas.width = cssWidth * dpr;
canvas.height = cssHeight * dpr;
ctx.scale(dpr, dpr);
kCanvas.width = kCssWidth * dpr;
kCanvas.height = kCssHeight * dpr;
kCtx.scale(dpr, dpr);

const colors = {
  psi_s: '#5f6368',
  psi_s_current: '#111111',
  psi_g: '#CB297B',
  waypoint: '#5a82ff',
};
const projectCache = new Map();

function drawAxes(W, H, pad) {
  ctx.strokeStyle = 'rgba(0,0,0,0.10)';
  ctx.lineWidth = 1;
  ctx.strokeRect(pad, pad, W - 2 * pad, H - 2 * pad);
}

function drawPolyline(points, project, color, width, alpha, upto = points.length - 1) {
  if (!points || points.length === 0) return;
  const last = Math.max(0, Math.min(upto, points.length - 1));
  ctx.save();
  ctx.strokeStyle = color;
  ctx.globalAlpha = alpha;
  ctx.lineWidth = width;
  ctx.beginPath();
  const p0 = project(points[0]);
  ctx.moveTo(p0[0], p0[1]);
  for (let i = 1; i <= last; i++) {
    const p = project(points[i]);
    ctx.lineTo(p[0], p[1]);
  }
  ctx.stroke();
  ctx.restore();
}

function drawCircle(pt, project, color, r, fill = true, lineWidth = 2) {
  const p = project(pt);
  ctx.save();
  ctx.beginPath();
  ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
  if (fill) {
    ctx.fillStyle = color;
    ctx.fill();
  } else {
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
  }
  ctx.restore();
}

function drawText(text, pt, project, color) {
  const p = project(pt);
  ctx.save();
  ctx.fillStyle = color;
  ctx.font = 'bold 16px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.fillText(text, p[0] + 12, p[1] - 12);
  ctx.restore();
}

function drawCandidateChain(seg, localIdx, project) {
  if (!seg.candidates || !seg.candidate_valid) return;
  const pts = seg.candidates[localIdx] || [];
  const valid = seg.candidate_valid[localIdx] || [];
  const chain = [];
  for (let i = 0; i < pts.length; i++) {
    if (valid[i]) chain.push({idx: i, pt: pts[i]});
  }
  if (chain.length === 0) return;

  ctx.save();
  ctx.strokeStyle = 'rgba(90, 90, 90, 0.45)';
  ctx.lineWidth = 1.3;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  let p = project(seg.psi_s[localIdx]);
  ctx.moveTo(p[0], p[1]);
  for (const item of chain) {
    p = project(item.pt);
    ctx.lineTo(p[0], p[1]);
  }
  p = project(seg.psi_g[localIdx]);
  ctx.lineTo(p[0], p[1]);
  ctx.stroke();
  ctx.restore();

  for (const item of chain) {
    drawCircle(item.pt, project, '#8a8a8a', 5, true);
    drawText(String(item.idx + 1), item.pt, project, '#6f6f6f');
  }
}

function activeViewer() {
  if (typeof viewer !== 'undefined') return viewer;
  return window.__braxLatentViewer || null;
}

function currentGlobalIndex() {
  const currentViewer = activeViewer();
  const duration = (currentViewer && currentViewer.animator && currentViewer.animator.duration) ? currentViewer.animator.duration : ((SYNC_DATA.total_frames - 1) * SYNC_DATA.frame_dt);
  const time = (currentViewer && currentViewer.animator) ? Math.min(currentViewer.animator.time, duration) : 0;
  const progress = duration > 0 ? (time / duration) : 0;
  const idx = Math.round(progress * (SYNC_DATA.total_frames - 1));
  return Math.max(0, Math.min(SYNC_DATA.total_frames - 1, idx));
}

function currentSegment(globalIdx) {
  for (let segIdx = 0; segIdx < SYNC_DATA.segments.length; segIdx++) {
    const seg = SYNC_DATA.segments[segIdx];
    if (globalIdx < seg.end) {
      return {segIdx, localIdx: globalIdx - seg.start, seg};
    }
  }
  const last = SYNC_DATA.segments[SYNC_DATA.segments.length - 1];
  return {segIdx: SYNC_DATA.segments.length - 1, localIdx: last.length - 1, seg: last};
}

function computeSegmentBounds(seg) {
  const allPoints = [];
  for (const key of ['psi_s', 'psi_g', 'waypoint']) {
    for (const pt of seg[key]) allPoints.push(pt);
  }
  if (seg.candidates && seg.candidate_valid) {
    for (let t = 0; t < seg.candidates.length; t++) {
      const pts = seg.candidates[t] || [];
      const valid = seg.candidate_valid[t] || [];
      for (let i = 0; i < pts.length; i++) {
        if (valid[i]) allPoints.push(pts[i]);
      }
    }
  }
  if (allPoints.length === 0) {
    return {centerX: 0, centerY: 0, span: 1};
  }
  const xs = allPoints.map(pt => pt[0]);
  const ys = allPoints.map(pt => pt[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  return {
    centerX: 0.5 * (minX + maxX),
    centerY: 0.5 * (minY + maxY),
    span: Math.max(maxX - minX, maxY - minY, 1e-6),
  };
}

function makeProject(segIdx, seg) {
  if (projectCache.has(segIdx)) return projectCache.get(segIdx);
  const pad = 38;
  const W = cssWidth;
  const H = cssHeight;
  const bounds = computeSegmentBounds(seg);
  const scale = Math.min(W - 2 * pad, H - 2 * pad) / bounds.span;
  const view = {
    pad,
    W,
    H,
    project: (pt) => [
      0.5 * W + (pt[0] - bounds.centerX) * scale,
      0.5 * H - (pt[1] - bounds.centerY) * scale,
    ],
  };
  projectCache.set(segIdx, view);
  return view;
}

function drawKDistribution(seg, localIdx, wpRefreshed) {
  const rawProbs = (seg.k_probs && seg.k_probs[localIdx]) || [];
  const probs = rawProbs.map(value => {
    const probability = Number(value);
    return Number.isFinite(probability) ? Math.max(0, Math.min(1, probability)) : 0;
  });
  const selectedK = Number(seg.planning_k[localIdx]);
  const W = kCssWidth;
  const H = kCssHeight;
  const left = 82;
  const right = 28;
  const top = 42;
  const bottom = 78;
  const plotW = W - left - right;
  const plotH = H - top - bottom;

  kCtx.clearRect(0, 0, W, H);
  kCtx.save();
  kCtx.font = '14px ui-monospace, SFMono-Regular, Menlo, monospace';
  kCtx.textBaseline = 'middle';

  for (let tick = 0; tick <= 4; tick++) {
    const probability = tick / 4;
    const y = top + (1 - probability) * plotH;
    kCtx.beginPath();
    kCtx.moveTo(left, y);
    kCtx.lineTo(W - right, y);
    kCtx.strokeStyle = tick === 0 ? 'rgba(0,0,0,0.35)' : 'rgba(0,0,0,0.10)';
    kCtx.lineWidth = 1;
    kCtx.stroke();
    kCtx.fillStyle = '#596168';
    kCtx.textAlign = 'right';
    kCtx.fillText(probability.toFixed(2), left - 10, y);
  }

  if (probs.length > 0) {
    const slotW = plotW / probs.length;
    const barW = Math.max(5, Math.min(72, slotW * 0.64));
    for (let k = 0; k < probs.length; k++) {
      const probability = probs[k];
      const barH = probability * plotH;
      const x = left + k * slotW + 0.5 * (slotW - barW);
      const y = top + plotH - barH;
      const selected = k === selectedK;

      kCtx.fillStyle = selected ? colors.waypoint : '#a6abb1';
      kCtx.fillRect(x, y, barW, barH);
      kCtx.strokeStyle = selected ? '#3159c9' : '#858b91';
      kCtx.lineWidth = selected ? 2.5 : 1;
      kCtx.strokeRect(x, y, barW, barH);

      kCtx.fillStyle = selected ? '#3159c9' : '#646a70';
      kCtx.textAlign = 'center';
      kCtx.textBaseline = 'bottom';
      kCtx.font = `${selected ? 'bold ' : ''}13px ui-monospace, SFMono-Regular, Menlo, monospace`;
      kCtx.fillText(probability.toFixed(3), x + 0.5 * barW, Math.max(top + 14, y - 7));

      kCtx.fillStyle = selected ? '#3159c9' : '#444b51';
      kCtx.textBaseline = 'top';
      kCtx.font = `${selected ? 'bold ' : ''}15px ui-monospace, SFMono-Regular, Menlo, monospace`;
      kCtx.fillText(String(k), x + 0.5 * barW, top + plotH + 12);
    }
  }

  kCtx.fillStyle = '#30383e';
  kCtx.textAlign = 'center';
  kCtx.textBaseline = 'bottom';
  kCtx.font = 'bold 16px ui-monospace, SFMono-Regular, Menlo, monospace';
  kCtx.fillText('K* index (0 ... Kmax - 1)', left + 0.5 * plotW, H - 12);
  kCtx.save();
  kCtx.translate(20, top + 0.5 * plotH);
  kCtx.rotate(-Math.PI / 2);
  kCtx.fillText('softmax probability', 0, 0);
  kCtx.restore();
  kCtx.restore();

  const selectedProbability = probs[selectedK] ?? 0;
  const temperature = SYNC_DATA.analytic_k_sample_temperature;
  kMeta.innerHTML = `Kmax (logit count): <b>${probs.length}</b> &nbsp; indices: <b>0–${Math.max(0, probs.length - 1)}</b><br>selected K*: <b>${selectedK}</b> &nbsp; probability: <b>${selectedProbability.toFixed(4)}</b><br>softmax temperature: <b>${temperature}</b> &nbsp; refresh: <b>${wpRefreshed ? 'yes' : 'no'}</b>`;
}

function drawFrame() {
  if (!SYNC_DATA.segments.length) return;
  const globalIdx = currentGlobalIndex();
  const {segIdx, localIdx, seg} = currentSegment(globalIdx);
  const view = makeProject(segIdx, seg);
  const project = view.project;

  ctx.clearRect(0, 0, view.W, view.H);
  drawAxes(view.W, view.H, view.pad);

  drawPolyline(seg.psi_s, project, colors.psi_s, 1.4, 0.18);
  drawPolyline(seg.psi_g, project, colors.psi_g, 1.4, 0.18);

  drawPolyline(seg.psi_s, project, colors.psi_s, 2.3, 0.90, localIdx);
  drawPolyline(seg.psi_g, project, colors.psi_g, 2.3, 0.80, localIdx);
  drawCandidateChain(seg, localIdx, project);

  drawCircle(seg.psi_s[localIdx], project, colors.psi_s_current, 7, true);
  drawCircle(seg.psi_g[localIdx], project, colors.psi_g, 13, false, 7);
  drawCircle(seg.waypoint[localIdx], project, colors.waypoint, 13, false, 7);
  drawText(`\u0069=${seg.planning_i[localIdx]}`, seg.waypoint[localIdx], project, colors.waypoint);

  const kStar = seg.planning_k[localIdx];
  const planningI = seg.planning_i[localIdx];
  const wpRefreshed = seg.wp_update[localIdx];
  meta.innerHTML = `env: <b>${SYNC_DATA.env_name}</b><br>rollout index: <b>${seg.traj_idx + 1}</b> / ${SYNC_DATA.segments.length} &nbsp; frame: <b>${localIdx + 1}</b> / ${seg.length}<br>K*: <b>${kStar}</b> &nbsp; \u0069: <b>${planningI}</b> &nbsp; waypoint refresh: <b>${wpRefreshed ? 'yes' : 'no'}</b>`;
  drawKDistribution(seg, localIdx, wpRefreshed);
  requestAnimationFrame(drawFrame);
}

requestAnimationFrame(drawFrame);
""".replace("__SYNC_DATA__", json.dumps(sync_data))

    if "</head>" in html_str:
        html_str = html_str.replace("</head>", style_html + "\n</head>", 1)

    body_anchor = "<body>\n\n    <div id=\"brax-viewer\"></div>"
    if body_anchor in html_str:
        html_str = html_str.replace(body_anchor, "<body>\n" + layout_html, 1)
    else:
        html_str = html_str.replace("<div id=\"brax-viewer\"></div>", layout_html, 1)

    injected_js = (
        "window.__braxLatentViewer = viewer;\n"
        "viewer.setDirty();\n"
        "      if (viewer.animator && viewer.animator.setLoop) { viewer.animator.setLoop(false); }\n"
        + sync_js
    )
    if "viewer.setDirty();" in html_str:
        html_str = html_str.replace("viewer.setDirty();", injected_js, 1)
    elif "var viewer = new Viewer(domElement, system);" in html_str:
        html_str = html_str.replace(
            "var viewer = new Viewer(domElement, system);",
            (
                "var viewer = new Viewer(domElement, system);\n"
                "      window.__braxLatentViewer = viewer;\n"
                "      if (viewer.setDirty) { viewer.setDirty(); }\n"
                "      if (viewer.animator && viewer.animator.setLoop) { viewer.animator.setLoop(false); }\n"
                + sync_js
            ),
            1,
        )
    else:
        html_str = html_str.replace(
            "</body>",
            "<script type=\"application/javascript\">\n" + sync_js + "\n</script>\n</body>",
            1,
        )
    return html_str


def render(make_policy, params, env, exp_dir, exp_name, num_steps, wandb_step=None,
           episode_length=1001):
    """Render eval rollouts cycling through goals.

    C_slerp policies with render metadata get a synchronized latent planner panel
    next to the Brax viewer.
    """
    policy = make_policy(params)
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)

    stateful_policy = hasattr(policy, 'init_state')
    policy_accepts_init = "init_state" in inspect.signature(policy).parameters

    if stateful_policy:
        @jax.jit
        def jit_policy(obs, rng, policy_state):
            return policy(obs, rng, init_state=policy_state)
    elif policy_accepts_init:
        @jax.jit
        def jit_policy(obs, rng, init_state):
            return policy(obs, rng, init_state=init_state)
    else:
        jit_policy = jax.jit(policy)

    steps_per_goal = episode_length - 1
    rollout = []

    key = jax.random.PRNGKey(seed=1)
    has_fixed_goals = hasattr(env, "possible_goals")
    n_rollouts = _visualization_rollout_count(env)
    can_render_latents = (
        stateful_policy
        and hasattr(policy, "render_latents")
    )
    latent_segments = []
    frame_cursor = 0

    for rollout_idx in range(n_rollouts):
        key, reset_key = jax.random.split(key)
        state = jit_env_reset(rng=reset_key)
        state, key, _ = _apply_visualization_start_goal(
            env,
            state,
            key,
            rollout_idx,
        )

        if stateful_policy:
            policy_state = policy.init_state
        elif policy_accepts_init:
            init_state = state.obs[:env.state_dim]

        raw_obs_list = []
        raw_wp_list = []
        raw_wpstep_list = []
        raw_k_list = []
        raw_i_list = []
        raw_candidate_list = []
        raw_candidate_mask_list = []
        raw_k_prob_list = []

        for _ in range(steps_per_goal):
            rollout.append(state.pipeline_state)
            if can_render_latents:
                raw_obs_list.append(state.obs)
                wp_counter_before = policy_state[0]

            key, subkey = jax.random.split(key)
            if stateful_policy:
                action, policy_state = jit_policy(state.obs[None], subkey, policy_state)
            elif policy_accepts_init:
                action, _ = jit_policy(state.obs[None], subkey, init_state[None])
            else:
                action, _ = jit_policy(state.obs[None], subkey)
            action = action[0]

            if can_render_latents:
                raw_wp_list.append(policy_state[1][0])
                raw_k_list.append(policy_state[2][0])
                if len(policy_state) > 3:
                    raw_i_list.append(policy_state[3][0])
                else:
                    raw_i_list.append(jnp.zeros((), dtype=jnp.int32))
                if len(policy_state) > 5:
                    raw_candidate_list.append(policy_state[4][0])
                    raw_candidate_mask_list.append(policy_state[5][0])
                if len(policy_state) > 6:
                    raw_k_prob_list.append(policy_state[6][0])
                raw_wpstep_list.append(bool(int(np.asarray(wp_counter_before[0])) % policy.control_horizon == 0))

            state = jit_env_step(state, action)

        if can_render_latents and raw_obs_list:
            seg_len = len(raw_obs_list)
            all_obs = jnp.stack(raw_obs_list)
            all_wp = jnp.stack(raw_wp_list)
            psi_s, psi_g = policy.render_latents(all_obs)

            psi_s_np = np.asarray(psi_s)
            psi_g_np = np.asarray(psi_g)
            planning_k = np.asarray(jnp.stack(raw_k_list), dtype=np.int32)
            planning_i = np.asarray(jnp.stack(raw_i_list), dtype=np.int32)
            if raw_candidate_list:
                candidate_valid = np.asarray(jnp.stack(raw_candidate_mask_list), dtype=bool)
                candidates = np.asarray(jnp.stack(raw_candidate_list))
            else:
                candidate_valid = np.zeros((seg_len, 0), dtype=bool)
                candidates = np.zeros(
                    (seg_len, 0, all_wp.shape[-1]), dtype=np.float32
                )
            if raw_k_prob_list:
                k_probs = np.asarray(jnp.stack(raw_k_prob_list), dtype=np.float32)
            else:
                # Backward-compatible fallback for stateful render policies
                # created before K probabilities became part of policy state.
                num_k = int(getattr(policy, "max_waypoint_depth", 1))
                k_probs = np.zeros((seg_len, num_k), dtype=np.float32)
                clipped_k = np.clip(planning_k, 0, num_k - 1)
                k_probs[np.arange(seg_len), clipped_k] = 1.0
            latent_segments.append(_project_latent_segment({
                "traj_idx": int(rollout_idx),
                "start": int(frame_cursor),
                "end": int(frame_cursor + seg_len),
                "length": int(seg_len),
                "psi_s": psi_s_np,
                "psi_g": psi_g_np,
                "waypoint": np.asarray(all_wp),
                "candidates": candidates,
                "candidate_valid": candidate_valid,
                "planning_k": planning_k,
                "planning_i": planning_i,
                "k_probs": k_probs,
                "wp_update": np.asarray(raw_wpstep_list, dtype=bool),
            }))
        frame_cursor += steps_per_goal

    html_str = html.render(env.sys.tree_replace({"opt.timestep": env.dt}), rollout, height=1024)

    if has_fixed_goals and (hasattr(env, "possible_starts") or hasattr(env, "start_modes")):
        html_str = _topdown_camera_patch(html_str, env)
    if latent_segments:
        html_str = _inject_latent_sync_panel(
            html_str,
            {
                "frame_dt": float(env.dt),
                "env_name": getattr(env, "env_name", env.__class__.__name__),
                "total_frames": len(rollout),
                "max_waypoint_depth": int(
                    getattr(policy, "max_waypoint_depth", 1)
                ),
                "analytic_k_sample_temperature": float(
                    getattr(
                        policy,
                        "analytic_k_eval_sample_temperature",
                        getattr(policy, "analytic_k_sample_temperature", 1.0),
                    )
                ),
                "segments": latent_segments,
            },
        )

    with open(os.path.join(exp_dir, f"{exp_name}_{num_steps}.html"), "w") as file:
        file.write(html_str)
    log_kwargs = {"render": wandb.Html(html_str)}
    if wandb_step is not None:
        wandb.log(log_kwargs, step=wandb_step)
    else:
        wandb.log(log_kwargs)


def capture_vis_rollouts(make_policy, params, env, episode_length,
                         critic_params, networks, state_size):
    """Run goal-cycling rollouts and return per-goal encoder representations.

    Same rollout logic as render() but only collects latent data for t-SNE
    visualization. Does not build HTML or log to wandb.
    """
    policy = make_policy(params)
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)

    stateful_policy = hasattr(policy, 'init_state')
    if stateful_policy:
        @jax.jit
        def jit_policy(obs, rng, policy_state):
            return policy(obs, rng, init_state=policy_state)
    else:
        jit_policy = jax.jit(policy)

    control_horizon = getattr(policy, "control_horizon", 1)
    n_rollouts = _visualization_rollout_count(env)
    steps_per_goal = episode_length - 1
    all_vis_data = []

    key = jax.random.PRNGKey(seed=1)
    for rollout_idx in range(n_rollouts):
        key, reset_key = jax.random.split(key)
        state = jit_env_reset(rng=reset_key)
        state, key, target = _apply_visualization_start_goal(
            env,
            state,
            key,
            rollout_idx,
        )

        if stateful_policy:
            policy_state = policy.init_state

        raw_obs_list = []
        raw_act_list = []
        raw_wp_list = []
        raw_wpstep_list = []
        raw_k_list = []

        for _ in range(steps_per_goal):
            if stateful_policy:
                wp_counter_before = policy_state[0]
            key, subkey = jax.random.split(key)
            if stateful_policy:
                action, policy_state = jit_policy(state.obs[None], subkey, policy_state)
            else:
                action, _ = jit_policy(state.obs[None], subkey)
            action = action[0]

            raw_obs_list.append(state.obs)
            raw_act_list.append(action)
            if stateful_policy:
                current_wp = policy_state[1]
                raw_wp_list.append(current_wp[0])
                is_wp_step = bool(int(wp_counter_before[0]) % control_horizon == 0)
                raw_wpstep_list.append(is_wp_step)
                if len(policy_state) > 2:
                    raw_k_list.append(policy_state[2][0])
            state = jit_env_step(state, action)

        all_obs = jnp.stack(raw_obs_list)
        all_act = jnp.stack(raw_act_list)
        states_raw = all_obs[:, :state_size]
        goals_raw = all_obs[:, state_size:]
        goal_indices = jnp.array(env.goal_indices)

        phi_s = networks["shared_encoder"].apply(critic_params["shared_encoder"], states_raw[:, goal_indices])
        psi_g = networks["shared_encoder"].apply(critic_params["shared_encoder"], goals_raw)
        phi_sa = networks["sa_encoder"].apply(
            critic_params["sa_encoder"],
            jnp.concatenate([states_raw, all_act], axis=-1))
        planning_k = None
        max_waypoint_depth = getattr(policy, "max_waypoint_depth", None)
        if raw_k_list:
            planning_k = jnp.stack(raw_k_list)
        elif max_waypoint_depth is not None and "cayley_v" in critic_params:
            from jaxgcrl.agents.cslerp.cslerp import (
                _cayley_rotation,
                select_analytic_k,
            )

            R = _cayley_rotation(critic_params["cayley_v"])
            planning_k, _ = select_analytic_k(
                phi_s,
                psi_g,
                R,
                max_waypoint_depth,
                selection="argmax",
            )

        vis_dict = {
            "phi_s": np.asarray(phi_s),
            "psi_g": np.asarray(psi_g),
            "phi_sa": np.asarray(phi_sa),
            "xy": np.asarray(states_raw[:, :2]),
            "goal_xy": np.asarray(target) if target is not None else np.asarray(goals_raw[0]),
        }
        if raw_wp_list:
            all_wp = jnp.stack(raw_wp_list)
            vis_dict["current_wp"] = np.asarray(all_wp)
            vis_dict["wp_step"] = np.array(raw_wpstep_list)
        if planning_k is not None:
            vis_dict["planning_k"] = np.asarray(planning_k)

        all_vis_data.append(vis_dict)

    return all_vis_data


def render_policy(params, save_path, env, actor, eval_env, vis_length):
    """Renders the policy and saves it as an HTML file."""

    # JIT compile the rollout function
    @jax.jit
    def policy_step(env_state, actor_params):
        means, _ = actor.apply(actor_params, env_state.obs)
        actions = nn.tanh(means)
        next_state = env.step(env_state, actions)
        return next_state, env_state  # Return current state for visualization

    rollout_states = []
    for i in range(10):
        env = create_env(eval_env) if type(eval_env) == str else eval_env

        # Initialize environment
        rng = jax.random.PRNGKey(seed=i + 1)
        env_state = jax.jit(env.reset)(rng)

        # Collect rollout using jitted function
        for _ in range(vis_length):
            env_state, current_state = policy_step(env_state, params)
            rollout_states.append(current_state.pipeline_state)

    # Render and save
    html_string = html.render(env.sys, rollout_states)
    render_path = f"{save_path}/vis.html"
    with open(render_path, "w") as f:
        f.write(html_string)
    wandb.log({"vis": wandb.Html(html_string)})

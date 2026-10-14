"""t-SNE visualization utilities for C_slerp representation space.

Pure CPU code (numpy, sklearn, matplotlib). Designed to run in a background
thread while training continues on GPU. -- rejected, because of GPU interaction.
eval code already computes the embeddings, so in this code, we just need to plot the embeddings.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Rectangle
from sklearn.manifold import TSNE

from jaxgcrl.envs.ant_maze import (
    BIG_MAZE,
    BIG_MAZE_EVAL,
    BIG_MAZE_SINGLE,
    EXTREME_MAZE,
    EXTREME_MAZE_EVAL,
    EXTREME_MAZE_V2,
    EXTREME_MAZE_V2_EVAL,
    GOAL,
    HARDEST_MAZE,
    HARDEST_MAZE_EVAL,
    RESET,
    SPIRAL_MAZE_EASY,
    SPIRAL_MAZE_EASY_EVAL,
    SPIRAL_MAZE_HARD,
    SPIRAL_MAZE_HARD_EVAL,
    U4_MAZE,
    U4_MAZE_EVAL,
    U4_MAZE_SINGLE,
    U5_MAZE,
    U5_MAZE_EVAL,
    U_MAZE,
    U_MAZE_EVAL,
)

MAZE_LAYOUTS = {
    "big_maze": BIG_MAZE,
    "big_maze_eval": BIG_MAZE_EVAL,
    "big_maze_single": BIG_MAZE_SINGLE,
    "u_maze": U_MAZE,
    "u_maze_eval": U_MAZE_EVAL,
    "u4_maze": U4_MAZE,
    "u4_maze_eval": U4_MAZE_EVAL,
    "u4_maze_single": U4_MAZE_SINGLE,
    "u5_maze": U5_MAZE,
    "u5_maze_eval": U5_MAZE_EVAL,
    "hardest_maze": HARDEST_MAZE,
    "hardest_maze_eval": HARDEST_MAZE_EVAL,
    "spiral_maze_easy": SPIRAL_MAZE_EASY,
    "spiral_maze_easy_eval": SPIRAL_MAZE_EASY_EVAL,
    "spiral_maze_hard": SPIRAL_MAZE_HARD,
    "spiral_maze_hard_eval": SPIRAL_MAZE_HARD_EVAL,
    "extreme_maze": EXTREME_MAZE,
    "extreme_maze_eval": EXTREME_MAZE_EVAL,
    "extreme_maze_v2": EXTREME_MAZE_V2,
    "extreme_maze_v2_eval": EXTREME_MAZE_V2_EVAL,
}

# Panel size (inches) for each subplot: grid, rollout, and maze use the same footprint.
_TSNE_PANEL_INCHES = (10, 10)

ROLLOUT_COLORS = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#ff7f0e",  # orange
    "#17becf",  # cyan
    "#e377c2",  # magenta
    "#8c564b",  # brown
    "#bcbd22",  # olive
    "#7f7f7f",  # gray
]


def _rollout_cmap(goal_idx):
    base = ROLLOUT_COLORS[goal_idx % len(ROLLOUT_COLORS)]
    return LinearSegmentedColormap.from_list(
        f"rollout_{goal_idx}", ["#f2f2f2", to_rgb(base)]
    )


def construct_grid_data(env, maze_layout_name, maze_size_scaling, sub_grid_res=1):
    """Build grid states and goals for all free cells in the maze.

    Args:
        env: unwrapped environment instance (needs sys.init_q).
        maze_layout_name: string key into MAZE_LAYOUTS.
        maze_size_scaling: float scaling factor for the maze cells.
        sub_grid_res: points per cell edge (1 = cell centers only, 3 = 3x3 sub-grid).

    Returns:
        grid_states: (N, state_dim-ish) ndarray — state-like values at each grid point.
        grid_goals:  (N, goal_dim) ndarray — goal for each grid point.
        grid_xy:     (N, 2) ndarray  — same as grid_goals (world xy).
    """
    layout = MAZE_LAYOUTS.get(maze_layout_name)
    if layout is None:
        logging.warning("Unknown maze layout %s, skipping grid construction", maze_layout_name)
        return None, None, None
    scaling = maze_size_scaling

    free_cells = []
    for i, row in enumerate(layout):
        for j, cell in enumerate(row):
            if cell != 1:
                free_cells.append((i * scaling, j * scaling))

    if sub_grid_res > 1:
        offsets = np.linspace(-1.0, 1.0, sub_grid_res)
        expanded = []
        for cx, cy in free_cells:
            for dx in offsets:
                for dy in offsets:
                    expanded.append((cx + dx, cy + dy))
        free_cells = expanded

    grid_xy = np.array(free_cells, dtype=np.float32)
    N = len(free_cells)

    init_q = np.asarray(env.sys.init_q)
    qpos_base = init_q[:-2]
    qpos_batch = np.tile(qpos_base, (N, 1)).astype(np.float32)
    qpos_batch[:, 0] = grid_xy[:, 0]
    qpos_batch[:, 1] = grid_xy[:, 1]
    qvel_batch = np.zeros((N, 14), dtype=np.float32)
    grid_states = np.concatenate([qpos_batch, qvel_batch], axis=-1)
    goal_dim = len(getattr(env, "goal_indices", (0, 1)))
    if goal_dim == 2:
        grid_goals = grid_xy.copy()
    elif goal_dim == 3:
        target_z = float(getattr(env, "target_z_coord", 1.25))
        grid_goals = np.concatenate(
            [grid_xy, np.full((N, 1), target_z, dtype=np.float32)],
            axis=-1,
        )
    else:
        logging.warning(
            "Unsupported goal dim %s for maze grid construction, skipping",
            goal_dim,
        )
        return None, None, None

    return grid_states, grid_goals, grid_xy


def fit_tsne(grid_phi, grid_psi, rollout_data, subsample=10, random_state=42):
    """Fit a single t-SNE on grid + rollout embeddings.

    Args:
        grid_phi: (N, D) — grid state encodings.
        grid_psi: (N, D) — grid goal encodings.
        rollout_data: list of dicts or dict with (T, G, D) arrays.
            Each dict has "phi_s" (T, D) arrays.
            Also "current_wp" (T, D) and "wp_step" (T,) bool.
            Also "psi_g" (T, D) — goal encoding (constant per rollout).
        subsample: take every N-th rollout step.
        random_state: for reproducibility.

    Returns:
        coords: (M, 2) t-SNE coordinates.
        slices: dict mapping group names to (start, end) index pairs.
    """
    parts = []
    slices = {}
    idx = 0

    def _add(name, arr):
        nonlocal idx
        parts.append(arr)
        slices[name] = (idx, idx + len(arr))
        idx += len(arr)

    _add("grid_phi", grid_phi)
    _add("grid_psi", grid_psi)

    if isinstance(rollout_data, list):
        goals_list = rollout_data
    else:
        G = rollout_data["phi_s"].shape[1]
        goals_list = [
            {k: v[:, g] if v.ndim > 1 else v for k, v in rollout_data.items()}
            for g in range(G)
        ]

    for gi, gd in enumerate(goals_list):
        phi_sub = gd["phi_s"][::subsample]
        _add(f"rollout_{gi}_phi", phi_sub)

        wp_mask = np.asarray(gd["wp_step"]).astype(bool)
        if wp_mask.any():
            wp_vecs = gd["current_wp"][wp_mask]
            _add(f"rollout_{gi}_wp", wp_vecs)

        goal_psi = gd["psi_g"][0:1] if gd["psi_g"].ndim >= 1 else gd["psi_g"][None]
        _add(f"rollout_{gi}_goal", goal_psi)

    all_emb = np.concatenate(parts, axis=0).astype(np.float32)
    M = len(all_emb)
    perplexity = min(50, max(5, M // 5))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
        max_iter=1000,
    )
    coords = tsne.fit_transform(all_emb)
    return coords, slices


def plot_grid_background(coords, slices, grid_xy, figsize=(10, 10), ax=None):
    """Plot 1: grid φ(s) in black, ψ(g) in gray, connected by lines."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    phi_s, phi_e = slices["grid_phi"]
    psi_s, psi_e = slices["grid_psi"]
    phi_pts = coords[phi_s:phi_e]
    psi_pts = coords[psi_s:psi_e]

    for i in range(len(phi_pts)):
        ax.plot(
            [phi_pts[i, 0], psi_pts[i, 0]],
            [phi_pts[i, 1], psi_pts[i, 1]],
            color="lightgray", alpha=0.4, linewidth=0.4, zorder=1,
        )

    ax.scatter(phi_pts[:, 0], phi_pts[:, 1], c="black", s=40, zorder=2, label="φ(s)")
    ax.scatter(psi_pts[:, 0], psi_pts[:, 1], c="gray", s=40, marker="^", zorder=2, label="ψ(g)")

    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Representation Structure (t-SNE)")
    ax.set_xticks([])
    ax.set_yticks([])
    if own_fig:
        fig.tight_layout()
    return fig


def plot_rollout_latent(coords, slices, rollout_data, grid_xy, figsize=(10, 10), ax=None):
    """Plot 2: faded grid + multi-rollout overlay with sequential colormaps."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    phi_s, phi_e = slices["grid_phi"]
    psi_s, psi_e = slices["grid_psi"]
    phi_pts = coords[phi_s:phi_e]
    psi_pts = coords[psi_s:psi_e]

    for i in range(len(phi_pts)):
        ax.plot(
            [phi_pts[i, 0], psi_pts[i, 0]],
            [phi_pts[i, 1], psi_pts[i, 1]],
            color="lightgray", alpha=0.08, linewidth=0.3, zorder=1,
        )
    ax.scatter(phi_pts[:, 0], phi_pts[:, 1], c="black", s=8, alpha=0.15, zorder=1)
    ax.scatter(psi_pts[:, 0], psi_pts[:, 1], c="gray", s=8, alpha=0.15, marker="^", zorder=1)

    if isinstance(rollout_data, list):
        n_goals = len(rollout_data)
    else:
        n_goals = rollout_data["phi_s"].shape[1]

    for gi in range(n_goals):
        cmap = _rollout_cmap(gi)
        phi_key = f"rollout_{gi}_phi"
        if phi_key not in slices:
            continue
        rs, re = slices[phi_key]
        pts = coords[rs:re]
        T = len(pts)
        t_norm = np.linspace(0.5, 1, T)
        colors = cmap(t_norm)

        ax.scatter(pts[:, 0], pts[:, 1], c=colors, s=6, zorder=3)
        ax.plot(pts[:, 0], pts[:, 1], color=cmap(0.5), alpha=0.3, linewidth=0.4, zorder=2)
        # Mark initial latent point for each rollout trajectory.
        ax.scatter(
            pts[0:1, 0],
            pts[0:1, 1],
            c=[cmap(0.2)],
            s=80,
            marker="s",
            edgecolors="black",
            linewidths=0.7,
            zorder=8,
        )

        wp_key = f"rollout_{gi}_wp"
        if wp_key in slices:
            ws, we = slices[wp_key]
            wp_pts = coords[ws:we]
            n_wp = len(wp_pts)
            for wi in range(n_wp):
                alpha = 0.3 + 0.7 * (wi / max(n_wp - 1, 1))
                ax.scatter(wp_pts[wi:wi+1, 0], wp_pts[wi:wi+1, 1],
                           c=[cmap(0.8)], s=150, marker="*",
                           edgecolors="black", linewidths=0.5,
                           alpha=alpha, zorder=5)
                ax.annotate(str(wi + 1),
                            (wp_pts[wi, 0], wp_pts[wi, 1]),
                            fontsize=6, fontweight="bold",
                            ha="center", va="bottom",
                            xytext=(0, 5), textcoords="offset points",
                            color=cmap(0.8), alpha=alpha, zorder=7)

        goal_key = f"rollout_{gi}_goal"
        if goal_key in slices:
            gs, ge = slices[goal_key]
            g_pt = coords[gs:ge]
            ax.scatter(g_pt[:, 0], g_pt[:, 1], c=[cmap(1.0)], s=200,
                       marker="X", edgecolors="black", linewidths=1.0, zorder=6)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="*", color="gray", markersize=12,
               linestyle="None", markeredgecolor="black", label="Waypoint (numbered)"),
        Line2D([0], [0], marker="X", color="gray", markersize=10,
               linestyle="None", markeredgecolor="black", label="Goal"),
        Line2D([0], [0], marker="s", color="gray", markersize=8,
               linestyle="None", markeredgecolor="black", label="Initial point"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              framealpha=0.8)

    ax.set_title("Rollout in Representation Space (t-SNE)")
    ax.set_xticks([])
    ax.set_yticks([])
    if own_fig:
        fig.tight_layout()
    return fig


def plot_maze_physical(rollout_data, maze_layout_name, maze_size_scaling, figsize=(10, 10), ax=None):
    """Plot 3: physical maze with ant xy trajectories overlaid."""
    own_fig = ax is None
    layout = MAZE_LAYOUTS.get(maze_layout_name)
    if layout is None:
        if own_fig:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.figure
        ax.text(0.5, 0.5, f"Unknown layout: {maze_layout_name}", ha="center")
        if own_fig:
            fig.tight_layout()
        return fig

    scaling = maze_size_scaling
    rows = len(layout)
    cols = len(layout[0])

    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    for i in range(rows):
        for j in range(cols):
            x0 = i * scaling - scaling / 2
            y0 = j * scaling - scaling / 2
            if layout[i][j] == 1:
                ax.add_patch(Rectangle((x0, y0), scaling, scaling,
                                       facecolor="#555555", edgecolor="#333333", linewidth=0.5))
            else:
                ax.add_patch(Rectangle((x0, y0), scaling, scaling,
                                       facecolor="white", edgecolor="#dddddd", linewidth=0.3))

    if isinstance(rollout_data, list):
        goals_list = rollout_data
    else:
        G = rollout_data["xy"].shape[1]
        goals_list = [
            {k: v[:, g] if v.ndim > 1 else v for k, v in rollout_data.items()}
            for g in range(G)
        ]

    n_goals = len(goals_list)
    for gi, gd in enumerate(goals_list):
        cmap = _rollout_cmap(gi)
        xy = np.asarray(gd["xy"])
        T = len(xy)
        t_norm = np.linspace(0, 1, T)

        for t in range(T - 1):
            ax.plot(
                [xy[t, 0], xy[t + 1, 0]],
                [xy[t, 1], xy[t + 1, 1]],
                color=cmap(t_norm[t]), linewidth=1.0, alpha=0.7,
            )

        goal_xy = np.asarray(gd["goal_xy"])
        ax.scatter(goal_xy[0], goal_xy[1], c=[cmap(1.0)], s=200,
                   marker="X", edgecolors="black", linewidths=1.0, zorder=10)
        ax.scatter(xy[0, 0], xy[0, 1], c="black", s=80, zorder=10)
        ax.scatter(xy[-1, 0], xy[-1, 1], c=[cmap(1.0)], s=90,
                   marker="s", edgecolors="black", linewidths=0.8, zorder=11)

        if "planning_k" in gd:
            k_values = np.asarray(gd["planning_k"]).reshape(-1)
            if len(k_values) > 0:
                if "wp_step" in gd:
                    wp_mask = np.asarray(gd["wp_step"]).astype(bool)[:len(k_values)]
                    k_indices = np.flatnonzero(wp_mask)
                else:
                    k_indices = np.arange(len(k_values))
                if len(k_indices) > 0:
                    k_at_updates = k_values[k_indices]
                    change_mask = np.concatenate(
                        [np.array([False]), k_at_updates[1:] != k_at_updates[:-1]]
                    )
                    label_indices = [int(i) for i in k_indices[change_mask]]

                    label_color = cmap(1.0)
                    for idx in label_indices:
                        idx = min(idx, T - 1, len(k_values) - 1)
                        ax.annotate(
                            f"K={int(k_values[idx])}",
                            (xy[idx, 0], xy[idx, 1]),
                            xytext=(6, 6),
                            textcoords="offset points",
                            color=label_color,
                            fontsize=7,
                            fontweight="bold",
                            ha="left",
                            va="center",
                            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.2),
                            zorder=13,
                        )

    ax.set_xlim(-scaling, rows * scaling)
    ax.set_ylim(-scaling, cols * scaling)
    ax.set_aspect("equal")
    ax.set_title("Physical Maze — Ground Truth")
    ax.set_xticks([])
    ax.set_yticks([])
    if own_fig:
        fig.tight_layout()
    return fig


def plot_tsne_row(
    coords,
    slices,
    rollout_data,
    grid_xy,
    maze_layout_name,
    maze_size_scaling,
):
    """One Figure: three equal vector panels side-by-side (sizes match standalone plots)."""
    w, h = _TSNE_PANEL_INCHES
    fig_w, fig_h = 3 * w, h
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.12},
    )
    plot_grid_background(coords, slices, grid_xy, ax=axes[0])
    plot_rollout_latent(coords, slices, rollout_data, grid_xy, ax=axes[1])
    plot_maze_physical(rollout_data, maze_layout_name, maze_size_scaling, ax=axes[2])
    fig.tight_layout()
    return fig


def prepare_tsne_data(eval_enc, eval_data, grid_phi, grid_psi, grid_xy,
                      state_size, control_horizon, n_eval_vis):
    """Transfer JAX arrays to numpy and build eval_rollouts_list + grid_np.

    Args:
        eval_enc: dict of JAX arrays from reencode_rollouts (phi_s, psi_g, phi_sa, means_wp).
        eval_data: Transition from evaluator (observation, action, discount, ...).
        grid_phi: JAX array from encode_grid_fn.
        grid_psi: JAX array from encode_grid_fn.
        grid_xy: numpy array from construct_grid_data.
        state_size: int, observation state dimension.
        control_horizon: int, waypoint gating period.
        n_eval_vis: int, number of eval envs to visualize.

    Returns:
        eval_rollouts_list: list of per-goal numpy dicts for t-SNE.
        grid_np: dict with "phi", "psi", "xy" numpy arrays.
    """
    eval_np = {k: np.asarray(v) for k, v in eval_enc.items()}
    eval_extras = getattr(eval_data, "extras", {})
    if isinstance(eval_extras, dict):
        if "current_waypoint" in eval_extras:
            eval_np["means_wp"] = np.asarray(
                eval_extras["current_waypoint"][:, :n_eval_vis]
            )
        if "planning_k" in eval_extras:
            eval_np["planning_k"] = np.asarray(
                eval_extras["planning_k"][:, :n_eval_vis]
            )
        if "wp_step" in eval_extras:
            eval_np["wp_step"] = np.asarray(
                eval_extras["wp_step"][:, :n_eval_vis]
            ).astype(bool)

    eval_obs_np = np.asarray(eval_data.observation[:, :n_eval_vis])
    eval_np["xy"] = eval_obs_np[:, :, :2]

    eval_goals_xy = eval_obs_np[0, :, state_size:]
    eval_discount_np = np.asarray(eval_data.discount[:, :n_eval_vis])

    eval_rollouts_list = []
    for g in range(n_eval_vis):
        T = eval_np["phi_s"].shape[0]
        done_g = eval_discount_np[:, g] < 0.5
        if "wp_step" in eval_np:
            wp_step_g = eval_np["wp_step"][:, g]
        else:
            wp_counter_g = np.zeros(T, dtype=np.int32)
            c = 0
            for t_i in range(T):
                wp_counter_g[t_i] = c
                if done_g[t_i]:
                    c = 0
                else:
                    c += 1
            wp_step_g = (wp_counter_g % control_horizon) == 0

        eval_rollouts_list.append({
            "phi_s": eval_np["phi_s"][:, g],
            "psi_g": eval_np["psi_g"][:, g],
            "phi_sa": eval_np["phi_sa"][:, g],
            "current_wp": eval_np["means_wp"][:, g],
            "wp_step": wp_step_g,
            "xy": eval_np["xy"][:, g],
            "goal_xy": eval_goals_xy[g],
        })
        if "planning_k" in eval_np:
            eval_rollouts_list[-1]["planning_k"] = eval_np["planning_k"][:, g]

    grid_np = {
        "phi": np.asarray(grid_phi),
        "psi": np.asarray(grid_psi),
        "xy": grid_xy,
    }

    return eval_rollouts_list, grid_np


def background_tsne_and_log(
    eval_np, vis_data, grid_np,
    maze_layout_name, maze_size_scaling,
    current_step,
):
    """Run t-SNE, log one combined figure per prefix (eval, vis) to wandb as tsne/{prefix}.

    Designed to be called from a background worker thread. -- rejected, because of GPU interaction
    """
    import wandb

    grid_phi = grid_np["phi"]
    grid_psi = grid_np["psi"]
    grid_xy = grid_np["xy"]

    figs = {}
    for prefix, rollout_data in [("eval", eval_np), ("vis", vis_data)]:
        try:
            coords, slices = fit_tsne(grid_phi, grid_psi, rollout_data)
            figs[f"tsne/{prefix}"] = plot_tsne_row(
                coords,
                slices,
                rollout_data,
                grid_xy,
                maze_layout_name,
                maze_size_scaling,
            )
        except Exception as e:
            logging.warning("t-SNE %s failed: %s", prefix, e)

    if figs:
        log_dict = {k: wandb.Image(v) for k, v in figs.items()}
        log_dict["tsne_step"] = current_step
        logging.info(
            "[WandBDebug] t-SNE log | step=%d | keys=%s",
            current_step,
            ",".join(sorted(log_dict.keys())),
        )
        wandb.log(log_dict, step=current_step)

    for fig in figs.values():
        plt.close(fig)

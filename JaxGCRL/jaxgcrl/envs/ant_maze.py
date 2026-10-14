import os
import random as _stdlib_random
import xml.etree.ElementTree as ET
from collections import deque

import jax
import mujoco
from brax import base, math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from jax import numpy as jnp

# This is based on original Ant environment from Brax
# https://github.com/google/brax/blob/main/brax/envs/ant.py
# Maze creation dapted from: https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/locomotion/maze_env.py

RESET = R = "r"
GOAL = G = "g"


# ---------------------------------------------------------------------------
# DFS procedural maze generator (ported from C_slerp maze_maker.py)
# ---------------------------------------------------------------------------
def _generate_dfs_maze(size=13, loop_prob=0.35, seed=12345678):
    """Generate a procedural DFS maze with random loops.

    Returns a size×size list-of-lists where 1=wall and 0=free.
    The center cell is guaranteed free (ant start position).
    """
    rng = _stdlib_random.Random(seed)

    maze = [[1] * size for _ in range(size)]

    def _neighbors(x, y):
        dirs = [(2, 0), (-2, 0), (0, 2), (0, -2)]
        rng.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 1 <= nx < size - 1 and 1 <= ny < size - 1:
                yield nx, ny, dx, dy

    # DFS carving
    stack = [(1, 1)]
    maze[1][1] = 0
    while stack:
        x, y = stack[-1]
        cand = [(nx, ny, dx, dy)
                for nx, ny, dx, dy in _neighbors(x, y)
                if maze[nx][ny] == 1]
        if cand:
            nx, ny, dx, dy = rng.choice(cand)
            maze[x + dx // 2][y + dy // 2] = 0
            maze[nx][ny] = 0
            stack.append((nx, ny))
        else:
            stack.pop()

    # Add random loops (avoid 2×2 open rooms)
    def _creates_room(mi, mj):
        for dx, dy in [(0, 0), (-1, 0), (0, -1), (-1, -1)]:
            ii, jj = mi + dx, mj + dy
            if 0 <= ii < size - 1 and 0 <= jj < size - 1:
                if (maze[ii][jj] == 0 and maze[ii + 1][jj] == 0
                        and maze[ii][jj + 1] == 0 and maze[ii + 1][jj + 1] == 0):
                    return True
        return False

    for i in range(1, size - 1):
        for j in range(1, size - 1):
            if maze[i][j] == 1 and rng.random() < loop_prob:
                maze[i][j] = 0
                if _creates_room(i, j):
                    maze[i][j] = 1

    maze[1][1] = 0
    maze[size - 2][size - 2] = 0
    center = size // 2
    maze[center][center] = 0

    return maze


def _snap_to_free(maze, target_i, target_j):
    """BFS from (target_i, target_j) to find the nearest free cell."""
    size = len(maze)
    if 0 <= target_i < size and 0 <= target_j < size and maze[target_i][target_j] == 0:
        return (target_i, target_j)
    visited = set()
    queue = deque([(target_i, target_j)])
    while queue:
        ci, cj = queue.popleft()
        if (ci, cj) in visited:
            continue
        visited.add((ci, cj))
        if 0 <= ci < size and 0 <= cj < size and maze[ci][cj] == 0:
            return (ci, cj)
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = ci + di, cj + dj
            if 0 <= ni < size and 0 <= nj < size and (ni, nj) not in visited:
                queue.append((ni, nj))
    return (target_i, target_j)


_DFS_MAZE_SIZE = 13
_DFS_MAZE_MATRIX = _generate_dfs_maze(size=_DFS_MAZE_SIZE)


U_MAZE = [
    [1, 1, 1, 1, 1],
    [1, R, G, G, 1],
    [1, 1, 1, G, 1],
    [1, G, G, G, 1],
    [1, 1, 1, 1, 1],
]

U_MAZE_EVAL = [
    [1, 1, 1, 1, 1],
    [1, R, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, G, G, G, 1],
    [1, 1, 1, 1, 1],
]

U4_MAZE = [[1, 1, 1, 1, 1],
           [1, G, G, G, 1],
           [1, R, 1, G, 1],
           [1, 1, 1, G, 1],
           [1, G, 1, G, 1],
           [1, G, G, G, 1],
           [1, 1, 1, 1, 1]]

U4_MAZE_EVAL = [[1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, R, 1, 0, 1],
                [1, 1, 1, 0, 1],
                [1, G, 1, 0, 1],
                [1, G, G, G, 1],
                [1, 1, 1, 1, 1]]

U4_MAZE_SINGLE = [[1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 1],
                  [1, R, 1, 0, 1],
                  [1, 1, 1, 0, 1],
                  [1, G, 1, 0, 1],
                  [1, 0, 0, 0, 1],
                  [1, 1, 1, 1, 1]]

U5_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1],
           [1, G, G, G, G, G, G, 1],
           [1, R, 1, 1, 1, 1, G, 1],
           [1, 1, 1, 1, 1, 1, G, 1],
           [1, G, 1, 1, 1, 1, G, 1],
           [1, G, G, G, G, G, G, 1],
           [1, 1, 1, 1, 1, 1, 1, 1]]

U5_MAZE_EVAL = [[1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, R, 1, 1, 1, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 0, 1],
                [1, G, 1, 1, 1, 1, G, 1],
                [1, G, G, G, G, G, G, 1],
                [1, 1, 1, 1, 1, 1, 1, 1]]

BIG_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, G, 1, 1, G, G, 1],
    [1, G, G, 1, G, G, G, 1],
    [1, 1, G, G, G, 1, 1, 1],
    [1, G, G, 1, G, G, G, 1],
    [1, G, 1, G, G, 1, G, 1],
    [1, G, G, G, 1, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

BIG_MAZE_EVAL = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 1, 1, G, G, 1],
    [1, 0, 0, 1, 0, G, G, 1],
    [1, 1, G, G, 0, 1, 1, 1],
    [1, 0, G, 1, 0, 0, 0, 1],
    [1, 0, 1, G, 0, 1, G, 1],
    [1, 0, G, G, 1, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

BIG_MAZE_SINGLE = [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, R, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, G, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
]

HARDEST_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, G, G, G, 1, G, G, G, G, G, 1],
    [1, G, 1, 1, G, 1, G, 1, G, 1, G, 1],
    [1, G, G, G, G, G, G, 1, G, G, G, 1],
    [1, G, 1, 1, 1, 1, G, 1, 1, 1, G, 1],
    [1, G, G, 1, G, 1, G, G, G, G, G, 1],
    [1, 1, G, 1, G, 1, G, 1, G, 1, 1, 1],
    [1, G, G, 1, G, G, G, 1, G, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]


HARDEST_MAZE_EVAL = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 0, G, 1, 0, 0, G, 0, G, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, G, 0, 0, 0, 0, G, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, G, 1, G, 1, 0, 0, G, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, G, 0, 1, G, 0, G, 1, 0, 0, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

EXTREME_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 1, G, G, G, 1, G, G, G, 1, G, G, G, 1, G, G, G, 1, G, G, G, 1, G, 1],
    [1, G, 1, G, 1, G, 1, G, 1, G, 1, G, 1, G, 1, G, 1, G, 1, G, 1, G, 1, G, 1],
    [1, G, G, G, 1, G, G, G, 1, G, G, G, 1, G, G, G, 1, G, G, G, 1, G, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]


EXTREME_MAZE_EVAL = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 1, G, 0, G, 1, G, 0, G, 1, G, 0, G, 1, G, 0, G, 1, G, 0, G, 1, G, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, G, 0, G, 1, G, 0, G, 1, G, 0, G, 1, G, 0, G, 1, G, 0, G, 1, G, 0, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

EXTREME_MAZE_V2 = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 1, G, G, G, 1, G, G, G, 1, G, G, G, 1, G, G, G, 1],
    [1, G, 1, G, 1, G, 1, G, 1, G, 1, G, 1, G, 1, G, 1, G, 1],
    [1, G, 1, G, 1, G, 1, G, 1, G, 1, G, 1, G, 1, G, 1, G, 1],
    [1, G, G, G, 1, G, G, G, 1, G, G, G, 1, G, G, G, 1, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]


EXTREME_MAZE_V2_EVAL = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 1, G, 0, G, 1, G, 0, G, 1, G, 0, G, 1, G, 0, G, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, G, 0, G, 1, G, 0, G, 1, G, 0, G, 1, G, 0, G, 1, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

SPIRAL_MAZE_EASY = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, R, G, G, G, G, 1],
    [1, 1, 1, 1, 1, G, 1],
    [1, 1, G, G, 1, G, 1],
    [1, 1, G, 1, 1, G, 1],
    [1, 1, G, G, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1],
]

SPIRAL_MAZE_EASY_EVAL = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 0, 1],
    [1, 1, G, G, 1, 0, 1],
    [1, 1, 0, 1, 1, 0, 1],
    [1, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
]

SPIRAL_MAZE_HARD = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, R, G, G, G, G, 1],
    [1, 1, 1, 1, 1, G, 1],
    [1, G, G, G, 1, G, 1],
    [1, G, 1, G, 1, G, 1],
    [1, G, 1, 1, 1, G, 1],
    [1, G, G, G, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1],
]

SPIRAL_MAZE_HARD_EVAL = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 0, 1],
    [1, 0, G, G, 1, 0, 1],
    [1, 0, 1, G, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
]


MAZE_HEIGHT = 0.5


def find_starts(structure, size_scaling):
    starts = []
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            if structure[i][j] == RESET:
                starts.append([i * size_scaling, j * size_scaling])

    return jnp.array(starts)


def find_goals(structure, size_scaling):
    goals = []
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            if structure[i][j] == GOAL:
                goals.append([i * size_scaling, j * size_scaling])

    return jnp.array(goals)


# Create a xml with maze and a list of possible goal positions
def make_maze(maze_layout_name, maze_size_scaling):
    is_dfs = maze_layout_name in ("dfs_maze", "dfs_maze_eval")

    if is_dfs:
        return _make_dfs_maze(maze_layout_name, maze_size_scaling)

    if maze_layout_name == "u_maze":
        maze_layout = U_MAZE
    elif maze_layout_name == "u_maze_eval":
        maze_layout = U_MAZE_EVAL
    elif maze_layout_name == "u4_maze":
        maze_layout = U4_MAZE
    elif maze_layout_name == "u4_maze_eval":
        maze_layout = U4_MAZE_EVAL
    elif maze_layout_name == "u4_maze_single":
        maze_layout = U4_MAZE_SINGLE
    elif maze_layout_name == "u5_maze":
        maze_layout = U5_MAZE
    elif maze_layout_name == "u5_maze_eval":
        maze_layout = U5_MAZE_EVAL
    elif maze_layout_name == "big_maze":
        maze_layout = BIG_MAZE
    elif maze_layout_name == "big_maze_eval":
        maze_layout = BIG_MAZE_EVAL
    elif maze_layout_name == "big_maze_single":
        maze_layout = BIG_MAZE_SINGLE
    elif maze_layout_name == "hardest_maze":
        maze_layout = HARDEST_MAZE
    elif maze_layout_name == "hardest_maze_eval":
        maze_layout = HARDEST_MAZE_EVAL
    elif maze_layout_name == "spiral_maze_easy":
        maze_layout = SPIRAL_MAZE_EASY
    elif maze_layout_name == "spiral_maze_easy_eval":
        maze_layout = SPIRAL_MAZE_EASY_EVAL
    elif maze_layout_name == "spiral_maze_hard":
        maze_layout = SPIRAL_MAZE_HARD
    elif maze_layout_name == "spiral_maze_hard_eval":
        maze_layout = SPIRAL_MAZE_HARD_EVAL
    elif maze_layout_name == "extreme_maze":
        maze_layout = EXTREME_MAZE
    elif maze_layout_name == "extreme_maze_eval":
        maze_layout = EXTREME_MAZE_EVAL
    elif maze_layout_name == "extreme_maze_v2":
        maze_layout = EXTREME_MAZE_V2
    elif maze_layout_name == "extreme_maze_v2_eval":
        maze_layout = EXTREME_MAZE_V2_EVAL
    else:
        raise ValueError(f"Unknown maze layout: {maze_layout_name}")

    xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "ant_maze.xml")

    possible_starts = find_starts(maze_layout, maze_size_scaling)
    possible_goals = find_goals(maze_layout, maze_size_scaling)

    tree = ET.parse(xml_path)
    worldbody = tree.find(".//worldbody")

    for i in range(len(maze_layout)):
        for j in range(len(maze_layout[0])):
            struct = maze_layout[i][j]
            if struct == 1:
                ET.SubElement(
                    worldbody,
                    "geom",
                    name="block_%d_%d" % (i, j),
                    pos="%f %f %f"
                    % (
                        i * maze_size_scaling,
                        j * maze_size_scaling,
                        MAZE_HEIGHT / 2 * maze_size_scaling,
                    ),
                    size="%f %f %f"
                    % (
                        0.5 * maze_size_scaling,
                        0.5 * maze_size_scaling,
                        MAZE_HEIGHT / 2 * maze_size_scaling,
                    ),
                    type="box",
                    material="",
                    contype="1",
                    conaffinity="1",
                    rgba="0.7 0.5 0.3 1.0",
                )

    tree = tree.getroot()
    xml_string = ET.tostring(tree)

    return xml_string, possible_starts, possible_goals


def _make_dfs_maze(maze_layout_name, maze_size_scaling):
    """Build XML and start/goal arrays for the procedural DFS maze.

    The maze is centered so that the middle cell maps to world (0, 0).
    """
    maze = _DFS_MAZE_MATRIX
    size = _DFS_MAZE_SIZE
    center = size // 2
    offset = -(size - 1) / 2.0 * maze_size_scaling

    # --- start: always center ---
    possible_starts = jnp.array([[0.0, 0.0]])

    # --- goals ---
    if maze_layout_name == "dfs_maze":
        goals = []
        for i in range(size):
            for j in range(size):
                if maze[i][j] == 0:
                    goals.append([
                        offset + i * maze_size_scaling,
                        offset + j * maze_size_scaling,
                    ])
        possible_goals = jnp.array(goals)
    else:
        # dfs_maze_eval: C_slerp eval goals scaled from cell_size=3 → current scaling.
        # C_slerp goals at cell_size=3 (offset=-18): [[-9,-12],[-12,9],[9,12],[12,-9],[0,0]]
        # Grid indices: (i,j) = round((pos + 18) / 3)
        eval_grid_cells = [(3, 2), (2, 9), (9, 10), (10, 3), (center, center)]
        goals = []
        for ei, ej in eval_grid_cells:
            si, sj = _snap_to_free(maze, ei, ej)
            goals.append([
                offset + si * maze_size_scaling,
                offset + sj * maze_size_scaling,
            ])
        possible_goals = jnp.array(goals)

    # --- build XML ---
    xml_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "assets", "ant_maze.xml"
    )
    tree = ET.parse(xml_path)
    worldbody = tree.find(".//worldbody")

    for i in range(size):
        for j in range(size):
            if maze[i][j] == 1:
                ET.SubElement(
                    worldbody,
                    "geom",
                    name="block_%d_%d" % (i, j),
                    pos="%f %f %f" % (
                        offset + i * maze_size_scaling,
                        offset + j * maze_size_scaling,
                        MAZE_HEIGHT / 2 * maze_size_scaling,
                    ),
                    size="%f %f %f" % (
                        0.5 * maze_size_scaling,
                        0.5 * maze_size_scaling,
                        MAZE_HEIGHT / 2 * maze_size_scaling,
                    ),
                    type="box",
                    material="",
                    contype="1",
                    conaffinity="1",
                    rgba="0.7 0.5 0.3 1.0",
                )

    tree = tree.getroot()
    xml_string = ET.tostring(tree)

    return xml_string, possible_starts, possible_goals


class AntMaze(PipelineEnv):
    def __init__(
        self,
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=False,
        backend="generalized",
        maze_layout_name="u_maze",
        maze_size_scaling=4.0,
        dense_reward: bool = False,
        center_bias_temp: float = 5.0,
        **kwargs,
    ):
        xml_string, possible_starts, possible_goals = make_maze(maze_layout_name, maze_size_scaling)

        sys = mjcf.loads(xml_string)
        self.possible_starts = possible_starts
        self.possible_goals = possible_goals

        # Center-biased goal sampling for DFS training maze
        if maze_layout_name == "dfs_maze":
            dists = jnp.linalg.norm(self.possible_goals, axis=1)
            w = jnp.exp(-dists / center_bias_temp)
            self.goal_weights = w / w.sum()
        else:
            self.goal_weights = None

        n_frames = 5

        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.005})
            n_frames = 10

        if backend == "mjx":
            sys = sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )

        if backend == "positional":
            # TODO: does the same actuator strength work as in spring
            sys = sys.replace(actuator=sys.actuator.replace(gear=200 * jnp.ones_like(sys.actuator.gear)))

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._use_contact_forces = use_contact_forces
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        self.dense_reward = dense_reward
        self.state_dim = 29
        self.goal_indices = jnp.array([0, 1])
        self.goal_reach_thresh = 0.5

        if self._use_contact_forces:
            raise NotImplementedError("use_contact_forces not implemented.")

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""

        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(rng, (self.sys.q_size(),), minval=low, maxval=hi)
        qd = hi * jax.random.normal(rng1, (self.sys.qd_size(),))

        # set the start and target q, qd
        start = self._random_start(rng2)
        q = q.at[:2].set(start)

        target = self._random_target(rng3)
        q = q.at[-2:].set(target)

        qd = qd.at[-2:].set(0)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        reward, done, zero = jnp.zeros(3)
        metrics = {
            "reward_forward": zero,
            "reward_survive": zero,
            "reward_ctrl": zero,
            "reward_contact": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "forward_reward": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero,
        }
        state = State(pipeline_state, obs, reward, done, metrics)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
        forward_reward = velocity[0]

        min_z, max_z = self._healthy_z_range
        is_healthy = jnp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jnp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))
        contact_cost = 0.0

        old_obs = self._get_obs(pipeline_state0)
        old_dist = jnp.linalg.norm(old_obs[:2] - old_obs[-2:])
        obs = self._get_obs(pipeline_state)
        dist = jnp.linalg.norm(obs[:2] - obs[-2:])
        vel_to_target = (old_dist - dist) / self.dt
        success = jnp.array(dist < self.goal_reach_thresh, dtype=float)
        success_easy = jnp.array(dist < 2.0, dtype=float)

        if self.dense_reward:
            reward = 10 * vel_to_target + healthy_reward - ctrl_cost - contact_cost
        else:
            reward = success

        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        state.metrics.update(
            reward_forward=forward_reward,
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            x_position=pipeline_state.x.pos[0, 0],
            y_position=pipeline_state.x.pos[0, 1],
            distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            forward_reward=forward_reward,
            dist=dist,
            success=success,
            success_easy=success_easy,
        )
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe ant body position and velocities."""
        qpos = pipeline_state.q[:-2]
        qvel = pipeline_state.qd[:-2]

        target_pos = pipeline_state.x.pos[-1][:2]

        if self._exclude_current_positions_from_observation:
            qpos = qpos[2:]

        return jnp.concatenate([qpos] + [qvel] + [target_pos])

    def _random_target(self, rng: jax.Array) -> jax.Array:
        """Returns a random target location.

        Uses center-biased weighted sampling for DFS training maze,
        uniform sampling for all other layouts.
        """
        idx = jax.random.choice(rng, len(self.possible_goals), p=self.goal_weights)
        return self.possible_goals[idx]

    def _random_start(self, rng: jax.Array) -> jax.Array:
        """Returns a random start location chosen from possibilities specified in the maze layout."""
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_starts))
        return jnp.array(self.possible_starts[idx])[0]

import re
import jax
import mujoco
import mujoco.mjx as mjx

from flax import struct
from etils import epath
from typing import Any, Dict, Optional

from builderbench.constants import _TIME_STEPS, _MJX_PARAMS

@struct.dataclass
class State:
    """Environment state for training and inference."""
    data: mjx.Data                                   # mjx data
    obs: jax.Array                                   # observation for the RL agent
    reward: jax.Array                                # reward for the RL agent
    done: jax.Array                                  # terminal signal for the RL agent
    metrics: Dict[str, jax.Array]                    # metrics to store during training 
    info: Dict[str, Any]                             # misc information

def mjx_make_data(
	model: mujoco.MjModel,
	qpos: Optional[jax.Array] = None,
	qvel: Optional[jax.Array] = None,
	ctrl: Optional[jax.Array] = None,
	act: Optional[jax.Array] = None,
	mocap_pos: Optional[jax.Array] = None,
	mocap_quat: Optional[jax.Array] = None,
	impl: Optional[str] = None,
	nconmax: Optional[int] = None,
	njmax: Optional[int] = None,
	device: Optional[jax.Device] = None,
) -> mjx.Data:
	"""Initialize MJX Data."""
	data = mjx.make_data(
		model, impl=impl, nconmax=nconmax, njmax=njmax, device=device
	)
	if qpos is not None:
		data = data.replace(qpos=qpos)
	if qvel is not None:
		data = data.replace(qvel=qvel)
	if ctrl is not None:
		data = data.replace(ctrl=ctrl)
	if act is not None:
		data = data.replace(act=act)
	if mocap_pos is not None:
		data = data.replace(mocap_pos=mocap_pos.reshape(model.nmocap, -1))
	if mocap_quat is not None:
		data = data.replace(mocap_quat=mocap_quat.reshape(model.nmocap, -1))
	return data

def mjx_step_data(
    model: mjx.Model,
    data: mjx.Data,
    action: jax.Array,
    n_substeps: int = 1,
) -> mjx.Data:
	def single_step(data, _):
		data = data.replace(ctrl=action)
		data = mjx.step(model, data)
		return data, None

	return jax.lax.scan(single_step, data, (), n_substeps)[0]


def make_env(args):
	# Initialize environment
	if re.fullmatch(r"creative-\d+-task\d+", args.env_id):
		num_cubes = int( re.search(r"creative-(\d+)", args.env_id).group(1))
		task_id = int( re.search(r"task(\d+)", args.env_id).group(1)) - 1

		from builderbench.creative_cube import CreativeCube, default_config
		env_class = CreativeCube
		default_config = default_config()
		default_config.num_cubes = num_cubes
		default_config.task_id = task_id
		episode_length = 100 + num_cubes * 50
	
	elif re.fullmatch(r"truncated-reward-creative-\d+-task\d+", args.env_id):
		num_cubes = int( re.search(r"creative-(\d+)", args.env_id).group(1))
		task_id = int( re.search(r"task(\d+)", args.env_id).group(1)) - 1

		from builderbench.creative_cube import TruncatedRewardCreativeCube, default_config
		env_class = TruncatedRewardCreativeCube
		default_config = default_config()
		default_config.num_cubes = num_cubes
		default_config.task_id = task_id
		episode_length = 100 + num_cubes * 50

	elif re.fullmatch(r"sparse-creative-\d+-task\d+", args.env_id):
		num_cubes = int( re.search(r"creative-(\d+)", args.env_id).group(1))
		task_id = int( re.search(r"task(\d+)", args.env_id).group(1)) - 1

		from builderbench.creative_cube import SparseCreativeCube, default_config
		env_class = SparseCreativeCube
		default_config = default_config()
		default_config.num_cubes = num_cubes
		default_config.task_id = task_id
		episode_length = 100 + num_cubes * 50

	elif re.fullmatch(r"planar-\d+-cube-\d+", args.env_id):
		halfgrid_size = int( re.search(r"planar-(\d+)", args.env_id).group(1))
		num_cubes = int( re.search(r"cube-(\d+)", args.env_id).group(1))

		assert num_cubes <= (2 * halfgrid_size) **2, "Number of cubes exceeds grid capacity"

		from builderbench.planar_cube import PlanarCube, default_config
		env_class = PlanarCube
		default_config = default_config()
		default_config.num_cubes = num_cubes
		default_config.halfgrid_size = halfgrid_size
		episode_length = 25 * num_cubes * halfgrid_size

	elif re.fullmatch(r"sparse-planar-\d+-cube-\d+", args.env_id):
		halfgrid_size = int( re.search(r"planar-(\d+)", args.env_id).group(1))
		num_cubes = int( re.search(r"cube-(\d+)", args.env_id).group(1))

		assert num_cubes <= (2 * halfgrid_size) **2, "Number of cubes exceeds grid capacity"

		from builderbench.planar_cube import SparsePlanarCube, default_config
		env_class = SparsePlanarCube
		default_config = default_config()
		default_config.num_cubes = num_cubes
		default_config.halfgrid_size = halfgrid_size
		episode_length = 25 * num_cubes * halfgrid_size
	
	elif re.fullmatch(r"planar-position-\d+-cube-\d+", args.env_id):
		halfgrid_size = int( re.search(r"planar-position-(\d+)", args.env_id).group(1))
		num_cubes = int( re.search(r"cube-(\d+)", args.env_id).group(1))

		assert num_cubes <= (2 * halfgrid_size) **2, "Number of cubes exceeds grid capacity"

		from builderbench.planar_position_cube import PlanarPositionCube, default_config
		env_class = PlanarPositionCube
		default_config = default_config()
		default_config.num_cubes = num_cubes
		default_config.halfgrid_size = halfgrid_size
		episode_length = 15 * ( num_cubes + 1) * halfgrid_size
	
	elif re.fullmatch(r"sparse-planar-position-\d+-cube-\d+", args.env_id):
		halfgrid_size = int( re.search(r"planar-position-(\d+)", args.env_id).group(1))
		num_cubes = int( re.search(r"cube-(\d+)", args.env_id).group(1))

		assert num_cubes <= (2 * halfgrid_size) **2, "Number of cubes exceeds grid capacity"

		from builderbench.planar_position_cube import SparsePlanarPositionCube, default_config
		env_class = SparsePlanarPositionCube
		default_config = default_config()
		default_config.num_cubes = num_cubes
		default_config.halfgrid_size = halfgrid_size
		episode_length = 15 * ( num_cubes ) * halfgrid_size
	
	elif re.fullmatch(r"discrete-planar-position-\d+-cube-\d+", args.env_id):
		halfgrid_size = int( re.search(r"planar-position-(\d+)", args.env_id).group(1))
		num_cubes = int( re.search(r"cube-(\d+)", args.env_id).group(1))

		assert num_cubes <= (2 * halfgrid_size) **2, "Number of cubes exceeds grid capacity"

		from builderbench.discrete_planar_position_cube import DiscretePlanarPositionCube, default_config
		env_class = DiscretePlanarPositionCube
		default_config = default_config()
		default_config.num_cubes = num_cubes
		default_config.halfgrid_size = halfgrid_size
		episode_length = 15 * ( num_cubes + 1) * halfgrid_size

	elif re.fullmatch(r"sparse-discrete-planar-position-\d+-cube-\d+", args.env_id):
		halfgrid_size = int( re.search(r"planar-position-(\d+)", args.env_id).group(1))
		num_cubes = int( re.search(r"cube-(\d+)", args.env_id).group(1))

		assert num_cubes <= (2 * halfgrid_size) **2, "Number of cubes exceeds grid capacity"

		from builderbench.discrete_planar_position_cube import SparseDiscretePlanarPositionCube, default_config
		env_class = SparseDiscretePlanarPositionCube
		default_config = default_config()
		default_config.num_cubes = num_cubes
		default_config.halfgrid_size = halfgrid_size
		episode_length = 15 * ( num_cubes ) * halfgrid_size

	elif re.fullmatch(r"gripper-creative-\d+-task\d+", args.env_id):
		num_cubes = int(re.search(r"creative-(\d+)", args.env_id).group(1))
		task_id = int(re.search(r"task(\d+)", args.env_id).group(1)) - 1

		from builderbench.gripper_cube import GripperCube, default_config
		env_class = GripperCube
		default_config = default_config()
		default_config.num_cubes = num_cubes
		default_config.task_id = task_id
		episode_length = 100 + num_cubes * 50

	elif re.fullmatch(r"truncated-reward-gripper-creative-\d+-task\d+", args.env_id):
		num_cubes = int(re.search(r"creative-(\d+)", args.env_id).group(1))
		task_id = int(re.search(r"task(\d+)", args.env_id).group(1)) - 1

		from builderbench.gripper_cube import TruncatedRewardGripperCube, default_config
		env_class = TruncatedRewardGripperCube
		default_config = default_config()
		default_config.num_cubes = num_cubes
		default_config.task_id = task_id
		episode_length = 100 + num_cubes * 50

	elif re.fullmatch(r"sparse-gripper-creative-\d+-task\d+", args.env_id):
		num_cubes = int(re.search(r"creative-(\d+)", args.env_id).group(1))
		task_id = int(re.search(r"task(\d+)", args.env_id).group(1)) - 1

		from builderbench.gripper_cube import SparseGripperCube, default_config
		env_class = SparseGripperCube
		default_config = default_config()
		default_config.num_cubes = num_cubes
		default_config.task_id = task_id
		episode_length = 100 + num_cubes * 50

	else:
		raise ValueError(f"Environment {args.env_id} not supported")

	default_config.episode_length = args.env_episode_length if args.env_episode_length is not None else episode_length
	default_config.sim_dt, default_config.ctrl_dt = _TIME_STEPS[args.env_id]
	default_config.env_early_termination = args.env_early_termination
	default_config.permutation_invariant_reward = args.permutation_invariant_reward
	default_config.permute_init_at_reset = getattr(args, "permute_init_at_reset", True)
	default_config.goal_noise_at_reset = getattr(args, "goal_noise_at_reset", True)
	# Procedural structure layouts (spacing set at hard reset; min=max → fixed s)
	_STRUCTURE_LAYOUTS = {
		"gripper-creative-4-task2": "pyramid_211",
	}
	if args.env_id in _STRUCTURE_LAYOUTS:
		default_config.goal_layout = _STRUCTURE_LAYOUTS[args.env_id]
		default_config.randomize_structure_spacing = True
	xyz_scale = getattr(args, "xyz_scale", None)
	if (
		xyz_scale is not None
		and hasattr(default_config, "action_scale")
		and hasattr(default_config.action_scale, "xyz_scale")
	):
		default_config.action_scale.xyz_scale = float(xyz_scale)
	default_config.nconmax, default_config.njmax = _MJX_PARAMS[args.env_id][0] * args.num_envs, _MJX_PARAMS[args.env_id][1] 
	return env_class, default_config
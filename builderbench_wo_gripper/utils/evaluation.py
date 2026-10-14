import time
import mujoco
import jax
import jax.numpy as jnp
import numpy as np

from typing import Callable

from utils.networks import ScannedRNN
from utils.wrapper import EvalWrapper

def get_trajectory(policy, env, key, unroll_length):
    """Roll out a single eval episode and return ONLY the per-step fields needed to
    render a video, stacked over time.

    Stacking the full env ``State`` (all mjx ``Data`` arrays) over ``unroll_length``
    steps is a large device allocation (multi-GiB on big envs like creative-8) even
    though rendering uses just qpos/qvel/mocap. We stack only those small fields, so
    the memory is ~seq_len × a few floats instead of ~seq_len × the whole State.
    Returns a dict of stacked arrays with shape ``[unroll_length, num_envs, ...]``.
    """
    @jax.jit
    def _get_trajectory(key):
        env_key, key = jax.random.split(key)
        eval_first_state = env.reset(jax.random.split(env_key, 1))

        def render_fields(s):
            fields = {
                "qpos": s.data.qpos,
                "qvel": s.data.qvel,
                "target_mocap_pos": s.info["target_mocap_pos"],
                "target_mocap_quat": s.info["target_mocap_quat"],
            }
            # Optional: only some envs expose a goal visibility mask.
            if "goal_mask" in s.info:
                fields["goal_mask"] = s.info["goal_mask"]
            return fields

        def f(carry, unused_t):
            env_state, key = carry
            key, next_key = jax.random.split(key)
            action, _ = policy(env_state.obs, env_state.info["target_goal"], key)
            next_env_state = env.step(env_state, action)

            return (next_env_state, next_key), render_fields(next_env_state)

        _, render_traj = jax.lax.scan(
                f,
                (eval_first_state, key),
                (),
                length=unroll_length
            )
        return render_traj

    return _get_trajectory(key)

def generate_unroll(env, env_state, policy, key, unroll_length):
    """Step the env for unroll_length steps and return final state."""
    def body(i, carry):
        env_state, key = carry
        key, next_key = jax.random.split(key)
        actions, _ = policy(env_state.obs, env_state.info["target_goal"], key)
        next_env_state = env.step(env_state, actions)
        return (next_env_state, next_key)

    final_state, _ = jax.lax.fori_loop(
        0, unroll_length, body, (env_state, key)
    )
    return final_state

def generate_rnn_unroll(env, env_state, policy, key, unroll_length, rnn_state_dim, num_envs):
    """Step the env for unroll_length steps and return final state."""
    
    rnn_state = ScannedRNN.initialize_carry(
        num_envs, rnn_state_dim,
    )
    def body(i, carry):
        env_state, rnn_state, key = carry
        key, next_key = jax.random.split(key)
        rnn_state, actions, _, _ = policy(rnn_state, env_state.obs, env_state.info['target_goal'], env_state.done, key)
        next_env_state = env.step(env_state, actions)
        return (next_env_state, rnn_state, next_key)

    final_state, _, _ = jax.lax.fori_loop(
        0, unroll_length, body, (env_state, rnn_state, key)
    )
    return final_state

class Evaluator:
    def __init__(
        self,
        eval_env,
        eval_policy_fn: Callable,
        num_eval_envs: int,
        episode_length: int,
        key: jax.Array,
        rnn: bool = False,
        rnn_state_dim: int = 256,
    ):
    
        self._key = key
        self._eval_walltime = 0.0

        eval_env = EvalWrapper(eval_env)

        def generate_eval_unroll(policy_params, key):
            reset_keys = jax.random.split(key, num_eval_envs)
            eval_first_state = eval_env.reset(reset_keys)
            if rnn:
                return generate_rnn_unroll(
                    eval_env,
                    eval_first_state,
                    eval_policy_fn(policy_params),
                    key,
                    unroll_length=episode_length,
                    rnn_state_dim=rnn_state_dim,
                    num_envs=num_eval_envs,
                )
        
            else:
                return generate_unroll(
                    eval_env,
                    eval_first_state,
                    eval_policy_fn(policy_params),
                    key,
                    unroll_length=episode_length,
                )
            
        self._generate_eval_unroll = jax.jit(generate_eval_unroll)
        self._steps_per_unroll = episode_length * num_eval_envs

    def run_evaluation(
        self,
        policy_params,
        training_metrics,
    ):
        """Run one epoch of evaluation."""
        self._key, unroll_key = jax.random.split(self._key)

        t = time.time()
        eval_state = self._generate_eval_unroll(policy_params, unroll_key)

        eval_metrics = eval_state.info['eval_metrics']
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        metrics = {}

        for name, value in eval_metrics.episode_metrics.items():

            metrics.update({
                f'eval/episode_{name}': (
                    np.mean(value)
                )
            })

            if name in ["success", "easy_success", "hard_success", "stacking"]:
                value_rate_means = np.mean(value > 0)
                metrics[f'eval/episode_{name}_rate'] = value_rate_means

        metrics['eval/avg_episode_length'] = np.mean(eval_metrics.episode_steps)
        metrics['eval/epoch_eval_time'] = epoch_eval_time
        metrics['eval/sps'] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        metrics = {
            'eval/walltime': self._eval_walltime,
            **training_metrics,
            **metrics,
        }

        return metrics
    
def get_video(env_id, inference_policy, video_env, video_key, episode_length):

    render_traj = get_trajectory(inference_policy, video_env, video_key, episode_length)
    # Pull the small per-step render fields to host so the device rollout buffers can
    # be freed before the (host-side) rendering loop.
    render_traj = jax.tree_util.tree_map(np.asarray, render_traj)

    if 'goal_mask' in render_traj:
        video_env.model.geom_rgba[ video_env._mocap_targets_geom, -1 ] = 0.2
        video_env.model.geom_rgba[ video_env._mocap_targets_geom[ ~render_traj['goal_mask'][0][0] ], -1] = 0

    video_images = []
    for i in range(episode_length):
        if i % 2 == 0:
            video_images.append(video_env.render_from_info(
                render_traj['qpos'][i][0],
                render_traj['qvel'][i][0],
                render_traj['target_mocap_pos'][i][0],
                render_traj['target_mocap_quat'][i][0],
            ))
    return video_images
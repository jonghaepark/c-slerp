import inspect
import time

import jax
import numpy as np
from brax import envs
from brax.training import acting
from brax.training.types import Metrics, PolicyParams


# TODO: make only single type of Evaluator
class Evaluator(acting.Evaluator):
    """
    Class for running evaluation epochs in a training process.

    This class extends the base acting.Evaluator and is designed to evaluate the
    performance of policies based on given parameters and training metrics.
    It provides detailed evaluations by aggregating various metrics over episodes
    and calculating statistical summaries such as mean, standard deviation,
    maximum, and minimum values.

    This is an evaluator that behaves in the exact same way as brax Evaluator,
    but additionally it aggregates metrics with max, min.
    It also logs in how many episodes there was any success.
    """

    def run_evaluation(
        self,
        policy_params: PolicyParams,
        training_metrics: Metrics,
        aggregate_episodes: bool = True,
    ) -> Metrics:
        """Run one epoch of evaluation."""
        self._key, unroll_key = jax.random.split(self._key)

        t = time.time()
        eval_state = self._generate_eval_unroll(policy_params, unroll_key)
        eval_metrics = eval_state.info["eval_metrics"]
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        metrics = {}
        aggregating_fns = [
            (np.mean, ""),
            (np.std, "_std"),
            (np.max, "_max"),
            (np.min, "_min"),
        ]

        for fn, suffix in aggregating_fns:
            metrics.update(
                {
                    f"eval/episode_{name}{suffix}": (fn(value) if aggregate_episodes else value)
                    for name, value in eval_metrics.episode_metrics.items()
                }
            )

        # We check in how many env there was at least one step where there was success
        if "success" in eval_metrics.episode_metrics:
            metrics["eval/episode_success_any"] = np.mean(eval_metrics.episode_metrics["success"] > 0.0)

        metrics["eval/avg_episode_length"] = np.mean(eval_metrics.episode_steps)
        metrics["eval/epoch_eval_time"] = epoch_eval_time
        metrics["eval/sps"] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        metrics = {"eval/walltime": self._eval_walltime, **training_metrics, **metrics}

        return metrics  # pytype: disable=bad-return-type  # jax-ndarray


def generate_unroll(
    actor_step,
    training_state,
    env,
    env_state,
    unroll_length,
    extra_fields=(),
    key=None,
):
    """Collect trajectories of given unroll_length."""

    if key is None:
        @jax.jit
        def f(carry, unused_t):
            state = carry
            nstate, transition = actor_step(
                training_state, env, state, extra_fields=extra_fields
            )
            return nstate, transition

        final_state, data = jax.lax.scan(f, env_state, (), length=unroll_length)
        return final_state, data

    @jax.jit
    def f(carry, unused_t):
        state, key = carry
        key, step_key = jax.random.split(key)
        nstate, transition = actor_step(
            training_state,
            env,
            state,
            extra_fields=extra_fields,
            key=step_key,
        )
        return (nstate, key), transition

    (final_state, _), data = jax.lax.scan(f, (env_state, key), (), length=unroll_length)
    return final_state, data


class ActorEvaluator:
    """Single GPU evaluator that evaluates an arbitrary actor function. Used by the CRL agent."""

    def __init__(
        self,
        actor_step,
        eval_env,
        num_eval_envs,
        episode_length,
        key,
    ):
        self._key = key
        self._eval_walltime = 0.0
        actor_step_accepts_key = "key" in inspect.signature(actor_step).parameters

        eval_env = envs.training.EvalWrapper(eval_env)

        def generate_eval_unroll(training_state, key):
            rollout_key = None
            if actor_step_accepts_key:
                key, rollout_key = jax.random.split(key)
            reset_keys = jax.random.split(key, num_eval_envs)
            eval_first_state = eval_env.reset(reset_keys)
            final_state, data = generate_unroll(
                actor_step,
                training_state,
                eval_env,
                eval_first_state,
                unroll_length=episode_length,
                key=rollout_key,
            )
            return final_state, data

        self._generate_eval_unroll = jax.jit(generate_eval_unroll)
        self._steps_per_unroll = episode_length * num_eval_envs

    def _run_single_eval(
        self,
        training_state,
        unroll_key,
        aggregate_episodes=True,
    ):
        """Run one eval unroll. Returns (metrics, data)."""
        prefix = "eval"

        t = time.time()
        eval_state, data = self._generate_eval_unroll(training_state, unroll_key)
        eval_metrics = eval_state.info["eval_metrics"]
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        metrics = {}
        aggregating_fns = [
            (np.mean, ""),
        ]

        for fn, suffix in aggregating_fns:
            metrics.update(
                {
                    f"{prefix}/episode_{name}{suffix}": (
                        fn(value) if aggregate_episodes else value
                    )
                    for name, value in eval_metrics.episode_metrics.items()
                }
            )

        if "success" in eval_metrics.episode_metrics:
            metrics[f"{prefix}/episode_success_any"] = np.mean(
                eval_metrics.episode_metrics["success"] > 0.0
            )

        metrics[f"{prefix}/avg_episode_length"] = np.mean(eval_metrics.episode_steps)
        metrics[f"{prefix}/epoch_eval_time"] = epoch_eval_time
        metrics[f"{prefix}/sps"] = self._steps_per_unroll / epoch_eval_time

        self._eval_walltime = self._eval_walltime + epoch_eval_time
        metrics[f"{prefix}/walltime"] = self._eval_walltime

        return metrics, data

    def _run_eval_common(self, training_state, training_metrics, aggregate_episodes=True):
        """Shared eval logic. Returns (metrics, data)."""
        self._key, unroll_key = jax.random.split(self._key)
        metrics, data = self._run_single_eval(
            training_state, unroll_key, aggregate_episodes
        )
        return {**training_metrics, **metrics}, data

    def run_evaluation(self, training_state, training_metrics, aggregate_episodes=True):
        """Run one epoch of evaluation. Returns metrics only."""
        metrics, _ = self._run_eval_common(training_state, training_metrics, aggregate_episodes)
        return metrics

    def run_evaluation_with_data(self, training_state, training_metrics, aggregate_episodes=True):
        """Run one epoch of evaluation. Returns (metrics, eval_data)."""
        return self._run_eval_common(training_state, training_metrics, aggregate_episodes)

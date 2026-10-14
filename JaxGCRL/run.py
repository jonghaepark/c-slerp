import logging
import os
import pickle
import pprint

import jax
# jax.config.update("jax_debug_nans", True)

import tyro
from brax.io import model
from datetime import datetime
import wandb
from jaxgcrl.utils.config import Config
from jaxgcrl.utils.env import MetricsRecorder, create_env


def main(config: Config):
    """Main function orchestrating the overall setup, initialization, and execution
    of training and evaluation processes. This function performs the following:
    1. Environment setup
    2. Directory creation for logging and checkpoints
    3. Training function creation
    4. Metrics recording
    5. Progress logging and monitoring
    6. Model saving and inference

    Creates the following directory structure:
        ./runs/
            run_{name}_s_{seed}/  # Run-specific directory
                args.pkl          # Saved command-line arguments
                ckpt/            # Model checkpoints
    Initializes wandb logging if enabled. Runs training with profiling and
    saves profiling results.
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s|  %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    info = {**vars(config.run), **vars(config.agent)}

    utd_ratio = (
        config.run.num_envs
        * config.run.episode_length
        * config.agent.train_step_multiplier
        / config.agent.batch_size
    ) / (config.run.num_envs * config.agent.unroll_length)
    info["utd_ratio"] = utd_ratio
    info["agent"] = type(config.agent).__name__

    logging.info("Arguments:\n%s", pprint.pformat(info))

    
    date_str = datetime.now().strftime("%m%d-%H%M%S")
    agent = config.agent
    try:
        exp_name = (
            f"for_paper_{date_str}_{config.run.env}_xy"
            f"_gamma{getattr(agent, 'discounting', '')}"
            f"_alphaH{getattr(agent, 'alpha_high_init', '')}"
            f"_alphaHLR{getattr(agent, 'alpha_high_lr', '')}"
            f"_entL{getattr(agent, 'target_entropy_low_coef', '')}"
            f"_entH{getattr(agent, 'target_entropy_high_coef', '')}"
        )
    except:
        exp_name = "run"

    wandb.init(
        project=config.run.wandb_project_name,
        group=config.run.wandb_group,
        name=exp_name,
        config=info,
        mode="online" if config.run.log_wandb else "disabled",
        entity="RHCRL"
    )

    env = create_env(env_name=config.run.env, backend=config.run.backend)
    if config.run.eval_env:
        eval_env = create_env(env_name=config.run.eval_env, backend=config.run.backend)
    else:
        eval_env = env

    if config.run.vis_env:
        vis_env = create_env(env_name=config.run.vis_env, backend=config.run.backend)
    elif config.run.eval_env:
        vis_env = eval_env
    else:
        vis_env = env

    vis_env_name = config.run.vis_env or config.run.eval_env or config.run.env
    if config.run.maze_layout_name is None and "maze" in vis_env_name:
        if "ant_ball" in vis_env_name:
            detected_layout = vis_env_name[9:]
        elif "ant" in vis_env_name:
            detected_layout = vis_env_name[4:]
        elif "humanoid" in vis_env_name:
            detected_layout = vis_env_name[9:]
        elif "simple" in vis_env_name:
            detected_layout = vis_env_name[7:]
        else:
            detected_layout = None
        if detected_layout:
            config = config.replace(
                run=config.run.replace(maze_layout_name=detected_layout)
            )

    os.makedirs("./runs", exist_ok=True)
    config = config.replace(run=config.run.replace(exp_name=exp_name))
    run_dir = f"./runs/run_{exp_name}_s_{config.run.seed}"
    ckpt_dir = run_dir + "/ckpt"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(run_dir + "/args.pkl", "wb") as f:
        pickle.dump(vars(config), f)

    eval_metric_names = [
        "episode_dist",
        "episode_reward",
        "episode_reward_ctrl",
        "episode_reward_dist",
        "episode_reward_near",
        "episode_reward_survive",
        "episode_success",
        "episode_success_any",
        "episode_success_easy",
        "episode_success_hard",
    ]
    metrics_to_collect = [
        f"eval/{name}" for name in eval_metric_names
    ] + [
        "training/actor_loss",
        "training/log_alpha",
        "training/alpha_loss",
        "training/critic_loss",
        "training/entropy",
        "training/sps",
        "training/low_critic_loss",
        "training/high_critic_loss",
        "training/low_actor_loss",
        "training/alpha_low_loss",
        "training/low_entropy",
        "training/log_alpha_low",
        "training/alpha_low",
        "alpha_high/alpha_high_loss",
        "alpha_high/high_entropy",
        "alpha_high/target_entropy_high",
        "alpha_high/log_alpha_high",
        "alpha_high/alpha_high",
        "training/planning_k_mean",
        "training/planning_K_one_based_mean",
        "training/planning_i_mean",
        "training/planning_i_max",
        "training/adaptive_reach_score_mean",
        "training/adaptive_direct_goal_fraction",
        "training/adaptive_fallback_fraction",
        "training/low_policy_q",
        "training/embed_uniformity",
        "training/cayley_identity_fro_norm",
        "training/cayley_skew_frobenius_norm",
        "training/cayley_skew_mean_square",
        "training/low_categorical_accuracy",
        "training/high_categorical_accuracy",
    ]

    metrics_recorder = MetricsRecorder(
        config.run.total_env_steps,
        metrics_to_collect,
        run_dir,
        exp_name,
        mode=config.run.wandb_mode if config.run.log_wandb else "disabled",
        episode_length=config.run.episode_length,
    )

    _, params, _ = config.agent.train_fn(
        train_env=env,
        eval_env=eval_env,
        vis_env=vis_env,
        config=config.run,
        progress_fn=metrics_recorder.progress,
    )
    model.save_params(ckpt_dir + "/final", params)


def cli():
    tyro.cli(
        main,
        config=(
            tyro.conf.OmitArgPrefixes,
            tyro.conf.OmitSubcommandPrefixes,
            tyro.conf.ConsolidateSubcommandArgs,
        ),
    )


if __name__ == "__main__":
    cli()

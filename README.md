# Hyperspherical Contrastive Representations Unlock Planning and Exploration
This repository provides the official implementation of **C-Slerp**. For more details, please refer to the [project page](https://jonghaepark.github.io/c-slerp/).

## Running the experiments

Our implementation builds on existing benchmark codebases. The locomotion tasks are based on JaxGCRL (Bortkiewicz et al., 2025). The manipulation tasks are based on BuilderBench (Ghugare et al., 2026) and its direct-force variant (Yan et al., 2026).

- JaxGCRL: [arXiv:2408.11052](https://arxiv.org/pdf/2408.11052), [code](https://github.com/michalbortkiewicz/jaxgcrl)
- BuilderBench: [arXiv:2510.06288](https://arxiv.org/abs/2510.06288), [code](https://github.com/rajghugare19/builderbench); direct-force variant: [OpenReview](https://openreview.net/forum?id=knI0KISsbw), [code](https://github.com/David-Yan1/builderbench)

From the repository root, create the two Conda environments. The `jaxgcrl` environment is used for `JaxGCRL`, while the `builderbench` environment is shared by `builderbench_gripper` and `builderbench_wo_gripper`.

```bash
conda env create -f conda/jaxgcrl.yml
conda env create -f conda/builderbench.yml
```

Each package contains a training script, `scripts/train_cslerp.sh`, which takes a task name as an argument. Run it from the package directory after activating the matching Conda environment.

### JaxGCRL

```bash
conda activate jaxgcrl
cd JaxGCRL
bash ./scripts/train_cslerp.sh ant_u5_maze
```

Tasks: `ant_big_maze`, `ant_u4_maze`, `ant_u5_maze`, `ant_extreme_maze`, `humanoid`, `humanoid_u_maze`.

### BuilderBench with a gripper

```bash
conda activate builderbench
cd builderbench_gripper
bash ./scripts/train_cslerp.sh 3-tower
```

Tasks: `3-tower`, `4-tower`, `4-pyramid`, `5-tower`.

### BuilderBench without a gripper

```bash
conda activate builderbench
cd builderbench_wo_gripper
bash ./scripts/train_cslerp.sh 2-tower
```

Tasks: `2-tower`, `8-tower`.



## License

New code written for C-Slerp is licensed under the MIT License.
Third-party code and assets are subject to their original licenses.
import jax
from jax import tree_util
import jax.numpy as jnp

def count_parameters(params):
    return sum(jnp.prod(jnp.array(p.shape)) for p in jax.tree_util.tree_leaves(params))

def print_pytree_leaf_shapes(tree):
    paths_leaves, _ = tree_util.tree_flatten_with_path(tree)

    print(f"{'Leaf path':35s} | {'Shape':>10s}")
    print("-" * 50)

    for path, leaf in paths_leaves:
        name = '/'.join(str(p) for p in path)
        shape = str(getattr(leaf, "shape", "N/A"))
        print(f"{name:35s} | {shape:>10s}")

    print("-" * 50)

def print_tree_shape_comparison(tree1, tree2):
    paths_leaves1, _ = tree_util.tree_flatten_with_path(tree1)
    paths_leaves2, _ = tree_util.tree_flatten_with_path(tree2)

    # Unpack paths and leaves
    paths1, leaves1 = zip(*paths_leaves1)
    paths2, leaves2 = zip(*paths_leaves2)

    same_structure = (paths1 == paths2)
    print("Same structure:", same_structure)
    print("-" * 60)
    print(f"{'Leaf path':35s} | {'Shape 1':>10s} | {'Shape 2':>10s} | Match")
    print("-" * 60)

    for path, leaf1, leaf2 in zip(paths1, leaves1, leaves2):
        name = '/'.join(str(p) for p in path)
        shape1 = str(leaf1.shape)
        shape2 = str(leaf2.shape)
        match = leaf1.shape == leaf2.shape
        print(f"{name:35s} | {shape1:>10s} | {shape2:>10s} | {'OK' if match else 'NO'}")

    print("-" * 60)
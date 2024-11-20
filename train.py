import functools
import os
import urllib.request
import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import time

# Disable future warnings.
import warnings

from prepare_md17_dataset import prepare_datasets

warnings.simplefilter(action="ignore", category=FutureWarning)


def mean_squared_loss(
    energy_prediction, energy_target, forces_prediction, forces_target, forces_weight
):
    energy_loss = jnp.mean(optax.l2_loss(energy_prediction, energy_target))
    forces_loss = jnp.mean(optax.l2_loss(forces_prediction, forces_target))
    return energy_loss + forces_weight * forces_loss


def mean_absolute_error(prediction, target):
    return jnp.mean(jnp.abs(prediction - target))


@functools.partial(jax.jit, static_argnames=("model_apply", "batch_size"))
def eval_step(model_apply, batch, batch_size, forces_weight, params):
    energy, forces = model_apply(
        params,
        atomic_numbers=batch["atomic_numbers"],
        positions=batch["positions"],
        dst_idx=batch["dst_idx"],
        src_idx=batch["src_idx"],
        batch_segments=batch["batch_segments"],
        batch_size=batch_size,
    )
    loss = mean_squared_loss(
        energy_prediction=energy,
        energy_target=batch["energy"],
        forces_prediction=forces,
        forces_target=batch["forces"],
        forces_weight=forces_weight,
    )
    energy_mae = mean_absolute_error(energy, batch["energy"])
    forces_mae = mean_absolute_error(forces, batch["forces"])
    return loss, energy_mae, forces_mae


def prepare_batches(key, data, batch_size):
    # Determine the number of training steps per epoch.
    data_size = len(data["energy"])
    steps_per_epoch = data_size // batch_size

    # Draw random permutations for fetching batches from the train data.
    perms = jax.random.permutation(key, data_size)
    perms = perms[
        : steps_per_epoch * batch_size
    ]  # Skip the last batch (if incomplete).
    perms = perms.reshape((steps_per_epoch, batch_size))

    # Prepare entries that are identical for each batch.
    num_atoms = len(data["atomic_numbers"])
    batch_segments = jnp.repeat(jnp.arange(batch_size), num_atoms)
    atomic_numbers = jnp.tile(data["atomic_numbers"], batch_size)
    offsets = jnp.arange(batch_size) * num_atoms
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
    dst_idx = (dst_idx + offsets[:, None]).reshape(-1)
    src_idx = (src_idx + offsets[:, None]).reshape(-1)

    # Assemble and return batches.
    return [
        dict(
            energy=data["energy"][perm],
            forces=data["forces"][perm].reshape(-1, 3),
            atomic_numbers=atomic_numbers,
            positions=data["positions"][perm].reshape(-1, 3),
            dst_idx=dst_idx,
            src_idx=src_idx,
            batch_segments=batch_segments,
        )
        for perm in perms
    ]


@functools.partial(
    jax.jit, static_argnames=("model_apply", "optimizer_update", "batch_size")
)
def train_step(
    model_apply, optimizer_update, batch, batch_size, forces_weight, opt_state, params
):
    def loss_fn(params):
        energy, forces = model_apply(
            params,
            atomic_numbers=batch["atomic_numbers"],
            positions=batch["positions"],
            dst_idx=batch["dst_idx"],
            src_idx=batch["src_idx"],
            batch_segments=batch["batch_segments"],
            batch_size=batch_size,
        )
        loss = mean_squared_loss(
            energy_prediction=energy,
            energy_target=batch["energy"],
            forces_prediction=forces,
            forces_target=batch["forces"],
            forces_weight=forces_weight,
        )
        return loss, (energy, forces)

    (loss, (energy, forces)), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer_update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    energy_mae = mean_absolute_error(energy, batch["energy"])
    forces_mae = mean_absolute_error(forces, batch["forces"])
    return params, opt_state, loss, energy_mae, forces_mae


def train_model(
    key,
    model,
    train_data,
    valid_data,
    num_epochs,
    learning_rate,
    forces_weight,
    batch_size,
):
    # Initialize model parameters and optimizer state.
    key, init_key = jax.random.split(key)
    optimizer = optax.adam(learning_rate)
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(
        len(train_data["atomic_numbers"])
    )
    params = model.init(
        init_key,
        atomic_numbers=train_data["atomic_numbers"],
        positions=train_data["positions"][0],
        dst_idx=dst_idx,
        src_idx=src_idx,
    )
    opt_state = optimizer.init(params)

    # Batches for the validation set need to be prepared only once.
    key, shuffle_key = jax.random.split(key)
    valid_batches = prepare_batches(shuffle_key, valid_data, batch_size)

    start_time = time.time()
    # Train for 'num_epochs' epochs.
    for epoch in range(1, num_epochs + 1):
        # Prepare batches.
        key, shuffle_key = jax.random.split(key)
        train_batches = prepare_batches(shuffle_key, train_data, batch_size)

        # Loop over train batches.
        train_loss = 0.0
        train_energy_mae = 0.0
        train_forces_mae = 0.0
        for i, batch in enumerate(train_batches):
            params, opt_state, loss, energy_mae, forces_mae = train_step(
                model_apply=model.apply,
                optimizer_update=optimizer.update,
                batch=batch,
                batch_size=batch_size,
                forces_weight=forces_weight,
                opt_state=opt_state,
                params=params,
            )
            train_loss += (loss - train_loss) / (i + 1)
            train_energy_mae += (energy_mae - train_energy_mae) / (i + 1)
            train_forces_mae += (forces_mae - train_forces_mae) / (i + 1)

        # Evaluate on validation set.
        valid_loss = 0.0
        valid_energy_mae = 0.0
        valid_forces_mae = 0.0
        for i, batch in enumerate(valid_batches):
            loss, energy_mae, forces_mae = eval_step(
                model_apply=model.apply,
                batch=batch,
                batch_size=batch_size,
                forces_weight=forces_weight,
                params=params,
            )
            valid_loss += (loss - valid_loss) / (i + 1)
            valid_energy_mae += (energy_mae - valid_energy_mae) / (i + 1)
            valid_forces_mae += (forces_mae - valid_forces_mae) / (i + 1)

        # Print progress.
        print(f"epoch: {epoch: 3d}                    train:   valid:")
        print(f"    loss [a.u.]             {train_loss : 8.3f} {valid_loss : 8.3f}")
        print(
            f"    energy mae [kcal/mol]   {train_energy_mae: 8.3f} {valid_energy_mae: 8.3f}"
        )
        print(
            f"    forces mae [kcal/mol/Å] {train_forces_mae: 8.3f} {valid_forces_mae: 8.3f}"
        )

    end_time = time.time()
    print(f"Training took {end_time - start_time} seconds.")

    # Return final model parameters.
    return params


class MessagePassingModel(nn.Module):
    features: int = 32
    max_degree: int = 2
    num_iterations: int = 3
    num_basis_functions: int = 8
    cutoff: float = 5.0
    max_atomic_number: int = 118  # This is overkill for most applications.

    def energy(
        self, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size
    ):
        # 1. Calculate displacement vectors.
        positions_dst = e3x.ops.gather_dst(positions, dst_idx=dst_idx)
        positions_src = e3x.ops.gather_src(positions, src_idx=src_idx)
        displacements = positions_src - positions_dst  # Shape (num_pairs, 3).

        # 2. Expand displacement vectors in basis functions.
        basis = e3x.nn.basis(  # Shape (num_pairs, 1, (max_degree+1)**2, num_basis_functions).
            displacements,
            num=self.num_basis_functions,
            max_degree=self.max_degree,
            radial_fn=e3x.nn.reciprocal_bernstein,
            cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff),
        )

        # 3. Embed atomic numbers in feature space, x has shape (num_atoms, 1, 1, features).
        x = e3x.nn.Embed(
            num_embeddings=self.max_atomic_number + 1, features=self.features
        )(atomic_numbers)

        # 4. Perform iterations (message-passing + atom-wise refinement).
        for i in range(self.num_iterations):
            # Message-pass.
            if i == self.num_iterations - 1:  # Final iteration.
                # Since we will only use scalar features after the final message-pass, we do not want to produce non-scalar
                # features for efficiency reasons.
                y = e3x.nn.MessagePass(max_degree=0, include_pseudotensors=False)(
                    x, basis, dst_idx=dst_idx, src_idx=src_idx
                )
                # After the final message pass, we can safely throw away all non-scalar features.
                x = e3x.nn.change_max_degree_or_type(
                    x, max_degree=0, include_pseudotensors=False
                )
            else:
                # In intermediate iterations, the message-pass should consider all possible coupling paths.
                y = e3x.nn.MessagePass()(x, basis, dst_idx=dst_idx, src_idx=src_idx)
            y = e3x.nn.add(x, y)

            # Atom-wise refinement MLP.
            y = e3x.nn.Dense(self.features)(y)
            y = e3x.nn.silu(y)
            y = e3x.nn.Dense(self.features, kernel_init=jax.nn.initializers.zeros)(y)

            # Residual connection.
            x = e3x.nn.add(x, y)

        # 5. Predict atomic energies with an ordinary dense layer.
        element_bias = self.param(
            "element_bias",
            lambda rng, shape: jnp.zeros(shape),
            (self.max_atomic_number + 1),
        )
        atomic_energies = nn.Dense(
            1, use_bias=False, kernel_init=jax.nn.initializers.zeros
        )(x)  # (..., Natoms, 1, 1, 1)
        atomic_energies = jnp.squeeze(
            atomic_energies, axis=(-1, -2, -3)
        )  # Squeeze last 3 dimensions.
        atomic_energies += element_bias[atomic_numbers]

        # 6. Sum atomic energies to obtain the total energy.
        energy = jax.ops.segment_sum(
            atomic_energies, segment_ids=batch_segments, num_segments=batch_size
        )

        # To be able to efficiently compute forces, our model should return a single output (instead of one for each
        # molecule in the batch). Fortunately, since all atomic contributions only influence the energy in their own
        # batch segment, we can simply sum the energy of all molecules in the batch to obtain a single proxy output
        # to differentiate.
        return -jnp.sum(
            energy
        ), energy  # Forces are the negative gradient, hence the minus sign.

    @nn.compact
    def __call__(
        self,
        atomic_numbers,
        positions,
        dst_idx,
        src_idx,
        batch_segments=None,
        batch_size=None,
    ):
        if batch_segments is None:
            batch_segments = jnp.zeros_like(atomic_numbers)
            batch_size = 1

        # Since we want to also predict forces, i.e. the gradient of the energy w.r.t. positions (argument 1), we use
        # jax.value_and_grad to create a function for predicting both energy and forces for us.
        energy_and_forces = jax.value_and_grad(self.energy, argnums=1, has_aux=True)
        (_, energy), forces = energy_and_forces(
            atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size
        )

        return energy, forces


if __name__ == "__main__":
    # Model hyperparameters.
    features = 32
    max_degree = 2
    num_iterations = 3
    num_basis_functions = 16
    cutoff = 5.0

    # Training hyperparameters.
    num_train = 900
    num_valid = 100
    num_epochs = 100
    learning_rate = 0.01
    forces_weight = 1.0
    batch_size = 10

    # Create PRNGKeys.
    data_key, train_key = jax.random.split(jax.random.PRNGKey(0), 2)

    # Draw training and validation sets.
    train_data, valid_data, _ = prepare_datasets(
        data_key, num_train=num_train, num_valid=num_valid
    )

    # Create and train model.
    message_passing_model = MessagePassingModel(
        features=features,
        max_degree=max_degree,
        num_iterations=num_iterations,
        num_basis_functions=num_basis_functions,
        cutoff=cutoff,
    )
    params = train_model(
        key=train_key,
        model=message_passing_model,
        train_data=train_data,
        valid_data=valid_data,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        forces_weight=forces_weight,
        batch_size=batch_size,
    )

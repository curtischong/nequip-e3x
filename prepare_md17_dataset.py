import os
import urllib.request
import numpy as np
import jax
import jax.numpy as jnp

data_dir = "datasets"
filename = "md17_ethanol.npz"
filepath = os.path.join(data_dir, filename)

def download_md17():
  if not os.path.exists(filepath):
    print(f"Downloading {filepath} (this may take a while)...")
    urllib.request.urlretrieve(url=f"http://www.quantum-machine.org/gdml/data/npz/{filename}", filename=filepath)


def prepare_datasets(key, num_train, num_valid):
  dataset = np.load(filepath)

  # Make sure that the dataset contains enough entries.
  num_data = len(dataset['E'])
  num_draw = num_train + num_valid
  if num_draw > num_data:
    raise RuntimeError(
      f'datasets only contains {num_data} points, requested num_train={num_train}, num_valid={num_valid}')

  # Randomly draw train and validation sets from dataset.
  choice = np.asarray(jax.random.choice(key, num_data, shape=(num_draw,), replace=False))
  train_choice = choice[:num_train]
  valid_choice = choice[num_train:]

  # Determine mean energy of the training set.
  mean_energy = np.mean(dataset['E'][train_choice])  # ~ -97000

  # Collect and return train and validation sets.
  train_data = dict(
    energy=jnp.asarray(dataset['E'][train_choice, 0] - mean_energy),
    forces=jnp.asarray(dataset['F'][train_choice]),
    atomic_numbers=jnp.asarray(dataset['z']),
    positions=jnp.asarray(dataset['R'][train_choice]),
  )
  valid_data = dict(
    energy=jnp.asarray(dataset['E'][valid_choice, 0] - mean_energy),
    forces=jnp.asarray(dataset['F'][valid_choice]),
    atomic_numbers=jnp.asarray(dataset['z']),
    positions=jnp.asarray(dataset['R'][valid_choice]),
  )
  return train_data, valid_data, mean_energy

if __name__ == "__main__":
  if not os.path.exists(data_dir):
    os.mkdir(data_dir)
import os
import urllib.request
# Download the dataset.

data_dir = "datasets"
if not os.path.exists(data_dir):
  os.mkdir(data_dir)

filename = "md17_ethanol.npz"
filepath = os.path.join(data_dir, filename)
if not os.path.exists(filepath):
  print(f"Downloading {filepath} (this may take a while)...")
  urllib.request.urlretrieve(url=f"http://www.quantum-machine.org/gdml/data/npz/{filename}", filename=filepath)
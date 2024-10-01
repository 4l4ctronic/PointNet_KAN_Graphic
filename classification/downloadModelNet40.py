import os
import urllib.request
import zipfile

# URL of the dataset
# url = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
url = "http://modelnet.cs.princeton.edu/ModelNet40.zip"

# Directory where the dataset will be downloaded
data_dir = "modelnet.zip"

# Download the dataset
urllib.request.urlretrieve(url, data_dir)

# Extract the dataset
with zipfile.ZipFile(data_dir, 'r') as zip_ref:
    zip_ref.extractall(os.path.dirname(data_dir))

# Path to the extracted dataset
DATA_DIR = os.path.join(os.path.dirname(data_dir), "ModelNet40")

print(f"Dataset extracted to {DATA_DIR}")

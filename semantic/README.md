
## Running the Semantic Segmentation Branch

**Step 1: Download the Data**

First, run the following script to download the Stanford 3D semantic parsing dataset:

```bash
python downloadS3DIS.py
```
Alternatively, you can download the dataset from the following link: <br>
https://www.kaggle.com/datasets/bhargavrko619/indoor3d-sem-seg-hdf5-data

To download the dataset containing all the points, use the following link:
https://cvg-data.inf.ethz.ch/s3dis/Stanford3dDataset_v1.2_Aligned_Version.zip

**Note**: In `semantic.py`, make sure to set the correct path for the variables `BASE_DIR` and `H5_SUBDIR`, which should point to the location where the Stanford 3D semantic parsing data is downloaded.

**Step 2: Run the Semantic Segmentation Script**

After downloading the data, run the classification script:

```bash
python semantic.py
```


## Running the Semantic Segmentation Branch

**Step 1: Download the Data**

First, run the following script to download the Stanford 3D semantic parsing dataset:

```bash
python downloadModelNet40.py
```

**Note**: In `semantic.py`, make sure to set the correct path for the variables `BASE_DIR` and `H5_SUBDIR`, which should point to the location where the Stanford 3D semantic parsing data is downloaded.

**Step 2: Run the Semantic Segmentation Script**

After downloading the data, run the classification script:

```bash
python semantic.py
```

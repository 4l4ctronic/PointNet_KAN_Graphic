
## Segmentation Guide

This guide will help you run the segmentation part of the project.

## Step 1: Download the ShapeNet Part Dataset ##

You can download the ShapeNet Part dataset by running the following script:

```bash
bash downloadShapeNetPart.sh
```

Alternatively, you can download the dataset from these sources:
- [Hugging Face](https://huggingface.co/)
- [ShapeNet](https://shapenet.org/)

## Step 2: Run the Training Script ##

After downloading the data, you can run the training script:

```bash
python train.py
```

**Note**: In `train.py`, ensure that the correct path to the data is set in the variable `hdf5_data_dir`, which should point to where the ShapeNet Part dataset is stored.

After training, the model will be saved in the current directory where `train.py` was executed.

## Step 3: Run the Testing Script ##

Once training is complete, you can test the model by running the testing script:

```bash
python test.py
```

**Note**: In `test.py`, make sure the correct path for the `hdf5_data_dir` variable is set, pointing to the location of the test data.

---

Make sure the required libraries and dependencies are installed as per the [Installation Guide](Installation_Guide.md).

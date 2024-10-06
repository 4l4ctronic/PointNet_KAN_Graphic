
## Overview
This folder contains several useful Python scripts designed to aid in the visualization and robustness testing of 3D part segmentation and classification models.

**`visualize.py`** is used for visualizing 3D part segmentation problems of objects in the test set using 2024 points. It helps to understand how well the model segments the parts of an object during inference. Ensure you load the saved model after training to visualize the results. In `visualize.py`, make sure the correct path for the `hdf5_data_dir` variable is set, pointing to the location of the test data. Ensure that the model trained on the dataset is loaded before running the scripts. This will allow proper visualization of the 3D part segmentation results.

**`visualizeFullCAD.py`** visualizes 3D part segmentation of objects in the test set using all available points. It provides a complete view of the object segmentation by not restricting the visualization to a subset of points. As with `visualize.py`, make sure to load the saved model after training to visualize the object’s segmentation. In `visualizeFullCAD.py`, make sure the correct path for the `hdf5_data_dir` variable is set, pointing to the location of the test data. Ensure that the model trained on the dataset is loaded before running the scripts. This will allow proper visualization of the 3D part segmentation results.

**`robustness.py`** tests the robustness of the model specifically for the classification problem. You can drop the number of points during testing (modify the value of `N2` in the code) to assess how the model performs when fewer points are available. Use this script to evaluate how the model behaves when objects have incomplete or reduced point sets in the test set. In `robustness.py`, make sure to set the correct path for the variable `direction`, which should point to the location where the ModelNet40 data is downloaded.

**`models_cls.py`** contains two types of models that resemble the classification branch of PointNet with MLP. These models are deep, with and without T-Nets, and are similar to the classification branch of PointNet (see Fig. 2 in [this paper](https://arxiv.org/pdf/1612.00593)).

**`models_seg.py`** contains two types of models that resemble the segmentation branch of PointNet with MLP. These models are deep, with and without T-Nets, and are similar to the segmentation branch of PointNet (see Fig. 9 in [this paper](https://arxiv.org/pdf/1612.00593)).



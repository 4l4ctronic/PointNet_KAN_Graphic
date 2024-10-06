
## Overview
This folder contains several useful Python scripts designed to aid in the visualization and robustness testing of 3D part segmentation and classification models.

### Files Description

**`visualize.py`** is used for visualizing 3D part segmentation problems of objects in the test set using 2024 points. It helps to understand how well the model segments the parts of an object during inference. Ensure you load the saved model after training to visualize the results.

**`visualizeFullCAD.py`** visualizes 3D part segmentation of objects in the test set using all available points. It provides a complete view of the object segmentation by not restricting the visualization to a subset of points. As with `visualize.py`, make sure to load the saved model after training to visualize the object’s segmentation.

**`robustness.py`** tests the robustness of the model specifically for the classification problem. You can drop the number of points during testing (modify the value of `N2` in the code) to assess how the model performs when fewer points are available. Use this script to evaluate how the model behaves when objects have incomplete or reduced point sets.


### Loading the Model
For both `visualize.py` and `visualizeFullCAD.py`, ensure that the model trained on the dataset is loaded before running the scripts. This will allow proper visualization of the 3D part segmentation results.

### Robustness Testing
In `robustness.py`, you can experiment with different values of `N2` to simulate a scenario where the number of points per object is reduced, and test how robust the classification model is under such conditions.


## Usage
- For 3D part segmentation visualizations:
  ```bash
  python visualize.py

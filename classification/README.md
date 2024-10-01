
## Running the Classification Branch

**Step 1: Download the Data**

First, run the following script to download the ModelNet40 dataset:

```bash
python downloadModelNet40.py
```

**Note**: In `classification.py`, make sure to set the correct path for the variable `direction`, which should point to the location where the ModelNet40 data is downloaded.

**Step 2: Run the Classification Script**

After downloading the data, run the classification script:

```bash
python classification.py
```

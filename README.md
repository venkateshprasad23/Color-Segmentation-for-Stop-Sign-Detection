# Color-Segmentation-for-Stop-Sign-Detection

ECE 276A Course Project - 1

## Description

The project detects stop signs using a Logistic Regression model.

### Code Organisation

```
stop_sign_detector.py   -- Main code which implements the function segment_image and get_bounding_box. Used by AutoGrader.
labeling.py             -- Code which uses RoiPoly to hand-label images for creating the training dataset. 
data_partition          -- Code to partition the dataset into the training and validation dataset.
training_code.py        -- Code used to train the discriminative model using Logistic Regression.
```

### Results
**Masked Image Output of a Test Image**
![Masked Image Output of a Test Image](/Results/18.png)

**Bounding Box Output of a Test Image**
![Bounding Box Image Output of a Test Image](/Results/18bb.png)

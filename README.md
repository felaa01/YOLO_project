# Tool Classification and Localization with YOLOv8

This project implements a real-time object detection system for the classification and localization of industrial tools using the **YOLOv8** architecture. It was developed as part of a Master's degree program and compares YOLO's performance against previously trained MobileNet and Vision Transformer (ViT) models.

## 📋 Overview

The model is trained to detect **6 classes** of industrial tools:

| Class         | Description        |
|---------------|--------------------|
| `Cutter`      | Cutting tool       |
| `Tongs`       | Gripping tool      |
| `Screws_box`  | Box of screws      |
| `Screwdriver` | Manual screwdriver |
| `Drill_bit`   | Drill bit          |
| `Allen_keys`  | Allen/hex key set  |

The dataset consists of **600 annotated images** (100 per class) from the *SingleDetectionTool* dataset, labeled in YOLO format using Label Studio.

## 📁 Repository Structure

```         
.
├── Clasification_localization_YOLO.ipynb   # Main notebook
├── data/
│   └── custom.yaml                         # YOLO dataset configuration file
└── runs/                                   # Created automatically after training
    └── detect/
        └── train/
            └── weights/
                └── best.pt                 # Best trained model weights
```

## ⚙️ Requirements

-   Download Raw dataset (images + YOLO labels) from: <https://drive.google.com/file/d/1j8g6RSWCt5vYVT1mZ_qetC52Mff8KnWl/view?usp=drive_link>
-   Python 3.8+
-   The following packages (install via pip):

``` bash
pip install ultralytics tensorflow scikit-learn pandas numpy matplotlib opencv-python seaborn lxml Pillow tensorflow
```

> **Note:** A GPU is strongly recommended for training. The notebook was originally developed on Google Colab with GPU acceleration. Training on CPU will be significantly slower.

## 🚀 Getting Started

1.  **Clone the repository:**

    ``` bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Place the dataset zip** inside the `data/` folder:

    ```         
    data/Label_studio.zip
    data/custom.yaml
    ```

3.  **Open and run the notebook:**

    ``` bash
    jupyter notebook Clasification_localization_YOLO.ipynb
    ```

4.  Run all cells in order. The notebook will:

    -   Extract and organize the dataset into `train / val / test` splits (70% / 15% / 15%)
    -   Train a YOLOv8n model for 100 epochs
    -   Evaluate the model with metrics, confusion matrix, and classification report
    -   Run inference and visualize predictions vs. ground truth boxes

## 📊 Results

The trained YOLOv8n model achieves **perfect classification accuracy** on the test set across all 6 classes, and obtains significantly lower bounding box MAE values compared to MobileNet and ViT baselines.

| Metric                  | Value           |
|-------------------------|-----------------|
| Classification Accuracy | 100% (test set) |
| Box MAE (xmin)          | 0.002           |
| Box MAE (ymin)          | 0.003           |
| Box MAE (xmax)          | 0.002           |
| Box MAE (ymax)          | 0.003           |

## 🔧 Configuration

The `custom.yaml` file defines the dataset paths and class names used by YOLO. If you change the folder structure, update the paths inside this file accordingly.

## 📝 Notes

-   This notebook was originally developed in **Google Colab** with data stored in Google Drive. All Drive-specific code has been replaced with local path equivalents.
-   The `RUNS_PATH` and `PATH` variables are defined in the *Paths* cell and can be adjusted to match your local setup.
-   The base YOLO weights (`yolov8n.pt`) are downloaded automatically by the `ultralytics` library on first run.

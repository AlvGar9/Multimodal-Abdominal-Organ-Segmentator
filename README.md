# A Comparative Study of Domain Adaptation Strategies for 3D Medical Image Segmentation

## Abstract

This project investigates and compares four distinct strategies for adapting a 3D U-Net model for multi-organ segmentation from a source domain of CT scans to a target domain of MRI scans, particularly in low-data scenarios. The methodologies evaluated are: **1)** a baseline approach of training from scratch on limited target data, **2)** fine-tuning a pre-trained source model, **3)** Domain-Adversarial Neural Networks (DANN), and **4)** an Unsupervised Domain Adaptation (UDA) method using multi-level feature alignment. The performance of each strategy is systematically evaluated across various quantities of available target data, and results are benchmarked on both in-domain (AMOS) and out-of-domain (CHAOS) test sets.

---

## Project Structure

The repository is organized into modules for preprocessing, training, and evaluation, ensuring a clean separation of concerns.

```
PROJECT_ROOT/
├── data/
│   ├── AMOS22/
│   └── CHAOS_Train_Sets/
├── models/
│   ├── best_model_ct_full.pth
│   └── ... (all other pre-trained .pth files)
├── preprocessed_data/
│   └── ... (output of the preprocessing script)
├── evaluation_results/
│   └── ... (output of the evaluation script)
├── scripts/
│   ├── run_preprocessing.sh
│   └── ...
├── src/
│   ├── preprocessing/
│   │   └── ... (preprocessing utilities)
│   ├── training/
│   │   └── ... (modules for standard training/fine-tuning)
│   ├── dann/
│   │   └── ... (modules for DANN)
│   ├── uda/
│   │   └── ... (modules for UDA with feature alignment)
│   │
│   ├── train.py
│   ├── finetune.py
│   ├── train_dann.py
│   ├── train_uda.py
│   ├── run_evaluation.py
│   ├── evaluate.py
│   └── visualize.py
│
├── requirements.txt
└── README.md
```

---

## 1. Installation and Setup

This project is developed in Python and makes use of various scientific computing and deep learning libraries. Development was conducted entirely on the Cirrus cluster at the University of Edinburgh. For dependency management, it is recommended to use a `conda` environment.

### **Step 1: Create a Conda Environment**

```bash
conda create -n thesis_env python=3.10
conda activate thesis_env
```

### **Step 2: Install Dependencies**

A single `requirements.txt` file is provided, consolidating all necessary packages.

```bash
pip install -r requirements.txt
```
The key dependencies include `torch`, `monai`, `scikit-learn`, `pandas`, `SimpleITK`, `nibabel`, and `scikit-image`.

---

## 2. Data Acquisition and Preparation

The project uses two public datasets. The ones used in this study can be easily found in Zenodo. Please follow the links to download the data.

* **AMOS 22:** [AMOS Challenge 22 Grand Challenge](https://zenodo.org/records/7155725)
* **CHAOS:** [CHAOS Challenge - MICCAI 2019](https://zenodo.org/records/3431873)

### **Step 1: Download and Organize Data**

1.  Create a directory named `data/` in the project root if not present.
2.  Download the zipped archives for both datasets.
3.  Place the downloaded `.zip` files into the `data/` directory.
4.  Unzip the files. Your `data/` directory should look like this:
    ```
    data/
    ├── AMOS22/
    │   ├── imagesTr/
    │   ├── labelsTr/
    │   └── ...
    └── CHAOS_Train_Sets/
        └── Train_Sets/
            ├── CT/
            └── MR/
    ```

### **Step 2: Run Preprocessing**

The preprocessing script performs isotropic resampling, resizing to a uniform shape (192x192x192), and filters the segmentation masks to include only the target organs (spleen, kidneys, liver).

To run the full preprocessing pipeline, execute the following command from the project root:

```bash
bash scripts/run_preprocessing.sh
```

This script will process both the AMOS and CHAOS datasets and save the results in a new directory named `preprocessed_data/`. This output is required for all subsequent training and evaluation steps.

---

## 3. Workflow: Reproducing Thesis Results

This section details the steps to reproduce the models and generate the evaluation results presented in the manuscript.

### **Step 1 (Optional): Reproducing the Pre-trained Models**

For convenience, all 17 pre-trained models (`.pth` files) used in the thesis are included in the `models/` directory. If you wish to reproduce them from scratch, you can use the training scripts as described below. Each script will save its output model to the project root directory. Each script allows for additional args to control the amount of training data used for MRI, as simluation of the data regimes. In the following examples, some arbitrary cases are shown, modify them accordingly to the desired data regime.

#### **Baseline & From-Scratch Training**

This script trains a standard U-Net from scratch. It is used to generate the CT baseline and the MRI models trained on limited data. 

```bash
# Train the CT baseline model (uses all CT training data)
python src/train.py --modality CT

# Train an MRI model on 5 samples
python src/train.py --modality MR --num_samples 5
```

#### **Fine-Tuning**

This script fine-tunes the pre-trained CT model on subsets of MRI data. It requires the baseline CT model (`best_model_ct_full.pth`) to be present.

```bash
# Fine-tune on 15 MRI samples
python src/finetune.py 15

# Fine-tune on the full set of MRI training samples
python src/finetune.py all
```

#### **Domain-Adversarial Training (DANN)**

This script trains a DANN model, adapting from the full CT dataset to subsets of the MRI dataset. It also requires the baseline CT model as a starting point for the U-Net weights.

```bash
# Train DANN using 30 MRI samples
python src/train_dann.py 30
```

#### **UDA with Feature Alignment**

This script trains the UDA model with multi-level feature alignment. It also requires the baseline CT model.

```bash
# Train UDA using 54 MRI samples
python src/train_uda.py 54
```

### **Step 2: Generating All Manuscript Results**

This is the final step to generate all quantitative results (CSV files), qualitative results (prediction images), and summary tables.

**Prerequisite:** Ensure all 17 pre-trained models are present in the `models/` directory. The evaluation script is configured to find them there.

To run the complete evaluation pipeline on all models and all test sets (CT AMOS, MR AMOS, MR CHAOS), execute the following command from the project root:

```bash
python src/run_evaluation.py
```

#### **Evaluation Outputs**

The script will create a directory named `evaluation_results/`. Inside, a sub-folder will be created for each of the 17 models. Each sub-folder will contain:
* **Raw and Summary CSVs:** Detailed per-patient and summarized metrics (Dice, IoU, Hausdorff Distance) for each of the three test sets.
* **Prediction Images:** Qualitative visualizations for the first few samples of each test set, including a detailed error map (True Positive, False Positive, False Negative).
* **Summary Text Files:** A human-readable summary of the performance metrics.

At the end of the run, a final `evaluation_summary.csv` file will be created in the root of `evaluation_results/`, consolidating the key performance metrics for all models, which can be used to generate the main tables in the thesis.

---

## Description of Key Files and Modules

* `src/preprocessing/`: Contains utilities for resampling and resizing medical images.
* `src/training/`: Contains shared modules for standard training, including configuration (`config.py`), data loading (`data.py`), model definition (`model.py`), and optimizer setup (`optimizer.py`).
* `src/dann/`: Contains modules specific to the DANN methodology, including the `DANNUNet` model and the specialized training engine.
* `src/uda/`: Contains modules specific to the feature-alignment UDA methodology, including the `MultiLevelUDANet`, the paired `UDACombinedDataset`, and the custom alignment loss.
* `src/train.py`: Main script for training models from scratch.
* `src/finetune.py`: Main script for fine-tuning a pre-trained model.
* `src/train_dann.py`: Main script for DANN experiments.
* `src/train_uda.py`: Main script for feature-alignment UDA experiments.
* `src/evaluate.py`: A module containing the core logic for running inference, calculating metrics, and generating visualizations.
* `src/visualize.py`: A module containing visualization functions, such as the t-SNE plotter.
* `src/run_evaluation.py`: The primary script to orchestrate the final evaluation of all models on all test sets, generating all thesis results.
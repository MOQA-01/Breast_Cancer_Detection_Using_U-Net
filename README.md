# Breast Cancer Detection Using U-Net

This project implements a deep learning model based on the U-Net architecture to perform semantic segmentation on breast cancer ultrasound images. The segmentation model identifies benign and malignant tumors in the images, aiding early and accurate diagnosis of breast cancer.

Table of Contents
	•	Project Overview
	•	Dataset
	•	Model Architecture
	•	Installation
	•	Usage
	•	Results
	•	Contributors

Project Overview

The primary goal of this project is to create a robust breast cancer segmentation model using U-Net, a convolutional neural network specialized for biomedical image segmentation. The model is trained and validated on ultrasound images, and it predicts segmentation masks for breast cancer tumors.

Key features:
	•	U-Net Architecture: A well-known model for biomedical image segmentation.
	•	Image Preprocessing: Images resized to 256x256 for efficient model performance.
	•	Binary Classification: Differentiates between benign and malignant categories.
	•	Optimization: Utilizes Adam optimizer with binary cross-entropy loss.
	•	Performance Evaluation: Includes accuracy as the evaluation metric.

Dataset

The dataset includes ultrasound images of breast cancer cases, categorized into:
[Download_Dataset](https://www.kaggle.com/datasets/moqa01/dataset-busi-with-gt)
	1.	Benign Tumors
	2.	Malignant Tumors

Each image is paired with a segmentation mask to highlight the tumor regions.

Dataset Preprocessing:
	•	Resized to 256x256 resolution.
	•	Visualized using matplotlib to confirm image-mask alignment.

Model Architecture

The U-Net architecture consists of:
	1.	Encoder Path: Captures contextual features using convolutional layers and down-sampling.
	2.	Bottleneck: Acts as a bridge between the encoder and decoder paths.
	3.	Decoder Path: Reconstructs the segmentation mask using up-sampling and skip connections.

A custom convolution block is implemented for better feature extraction.

Installation
	1.	Clone the repository:

git clone https://github.com/MOQA-0/Breast-Cancer-Segmentation.git
cd Breast-Cancer-Segmentation


	2.	Install dependencies:

pip install -r requirements.txt


	3.	Download the dataset (link to be added) and place it in the /data directory.

Usage
	1.	Open the Jupyter Notebook:

jupyter notebook Breast_cancer-segmentation.ipynb


	2.	Run the notebook to:
	•	Load and preprocess the dataset.
	•	Define and train the U-Net model.
	•	Visualize predictions on validation/test data.
	3.	Use the provided .h5 model file for inference:

from keras.models import load_model
model = load_model('Breast Cancer Segmentation.h5')

Results

Training Performance:
	•	Loss Function: Binary cross-entropy.
	•	Evaluation Metric: Accuracy.

Visualizations:
	•	Ground Truth vs Predicted Segmentation Masks
	•	Tumor Regions Highlighted in Ultrasound Images

Contributors
	•	Mohammed Qalandar - Project Lead
	•	[Email](moqa-is@outlook.com)
 	•	[LinkedIn](https://www.linkedin.com/in/mohammed-qalandar-shah-quazi-b59428259/)
 


# Breast Cancer Detection Using U-Net

This project implements a deep learning model based on the U-Net architecture to perform semantic segmentation on breast cancer ultrasound images. The segmentation model identifies benign and malignant tumors in the images, aiding early and accurate diagnosis of breast cancer.

Table of Contents
	•	Project Overview<br>
	•	Dataset<br>
	•	Model Architecture<br>
	•	Installation<br>
	•	Usage<br>
	•	Results<br>
	•	Contributors<br>

Project Overview

The primary goal of this project is to create a robust breast cancer segmentation model using U-Net, a convolutional neural network specialized for biomedical image segmentation. The model is trained and validated on ultrasound images, and it predicts segmentation masks for breast cancer tumors.

Key features:<br>
	•	U-Net Architecture: A well-known model for biomedical image segmentation.<br>
	•	Image Preprocessing: Images resized to 256x256 for efficient model performance.<br>
	•	Binary Classification: Differentiates between benign and malignant categories.<br>
	•	Optimization: Utilizes Adam optimizer with binary cross-entropy loss.<br>
	•	Performance Evaluation: Includes accuracy as the evaluation metric.<br>

Dataset

The dataset includes ultrasound images of breast cancer cases, categorized into:<br>
[Download_Dataset](https://www.kaggle.com/datasets/moqa01/dataset-busi-with-gt)<br>
	1.	Benign Tumors<br>
	2.	Malignant Tumors<br>

Each image is paired with a segmentation mask to highlight the tumor regions.<br>

Dataset Preprocessing:<br>
	•	Resized to 256x256 resolution.<br>
	•	Visualized using matplotlib to confirm image-mask alignment.<br>

Model Architecture<br>

The U-Net architecture consists of:<br>
	1.	Encoder Path: Captures contextual features using convolutional layers and down-sampling.<br>
	2.	Bottleneck: Acts as a bridge between the encoder and decoder paths.<br>
	3.	Decoder Path: Reconstructs the segmentation mask using up-sampling and skip connections.<br>

A custom convolution block is implemented for better feature extraction.<br>

Installation<br>
	1.	Clone the repository:<br>

git clone https://github.com/MOQA-01/Breast-Cancer-Segmentation.git
cd Breast-Cancer-Segmentation


	2.	Install dependencies:

pip install -r requirements.txt


	3.	Download the dataset (link to be added) and place it in the /data directory.

Usage<br>
1.	Open the Jupyter Notebook:

jupyter notebook Breast_cancer-segmentation.ipynb


2.	Run the notebook to:<br>
	•	Load and preprocess the dataset.<br>
	•	Define and train the U-Net model.<br>
	•	Visualize predictions on validation/test data.<br>
 
3.	Use the provided .h5 model file for inference:<br>

from keras.models import load_model
model = load_model('Breast Cancer Segmentation.h5')

Results<br>

Training Performance:<br>
	•	Loss Function: Binary cross-entropy.<br>
	•	Evaluation Metric: Accuracy.<br>

Visualizations:<br>
	•	Ground Truth vs Predicted Segmentation Masks<br>
	•	Tumor Regions Highlighted in Ultrasound Images<br>

Contributors
	•	Mohammed Qalandar - Project Lead<br>
	•	[Email](moqa-is@outlook.com)<br>
 	•	[LinkedIn](https://www.linkedin.com/in/mohammed-qalandar-shah-quazi-b59428259/)
 


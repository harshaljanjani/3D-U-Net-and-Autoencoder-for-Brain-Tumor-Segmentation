# Brain Tumor Segmentation using Deep Learning

## Overview

This project focuses on automated segmentation of brain tumors from MRI scans using deep learning techniques. It utilizes the Brain Tumor Segmentation (BraTS) Challenge dataset, which contains multimodal MRI scans, specifically targeting the segmentation of gliomas, a prevalent type of brain tumor. The main goal is to develop a model that can accurately identify and segment tumor regions to assist in diagnosis, treatment planning, and monitoring.

## Problem Statement

Manual segmentation of brain tumors from MRI scans is a challenging and time-consuming process, often prone to human error. This project aims to automate the segmentation process using a deep learning model, leveraging modern neural network architectures like U-Net and 3D U-Net for precise delineation of tumor boundaries.

## Dataset

The project uses the BraTS dataset, which includes:

- **MRI Modalities**:

  - T1-weighted (T1): Basic anatomical imaging.
  - T1-weighted with contrast enhancement (T1Gd/T1ce): Highlights active tumor regions.
  - T2-weighted (T2): Shows tumor and surrounding edema.
  - FLAIR (Fluid Attenuated Inversion Recovery): Emphasizes lesions by suppressing fluid.

- **Segmentation Labels**:

  - Enhancing Tumor (ET): Actively growing part of the tumor.
  - Tumor Core (TC): Includes both enhancing and necrotic regions.
  - Whole Tumor (WT): Encompasses the entire tumor area, including the core and surrounding edema.

- **Data Format**:
  - The scans are provided as 3D volumes in the NIfTI (.nii) format, with dimensions typically of 240x240x155 voxels.

## Model Architectures

### 3D U-Net

The primary model used is a 3D U-Net, designed for volumetric data like MRI scans. It follows an encoder-decoder structure with skip connections, enabling the model to capture both global context and fine details.

- **Encoder**: Extracts spatial features through 3D convolutions and downsampling.
- **Decoder**: Reconstructs the segmented regions using transposed convolutions, aided by skip connections.
- **Skip Connections**: Preserve high-resolution features by passing them directly from the encoder to the decoder, enhancing the segmentation accuracy.

### 3D Autoencoder

A 3D autoencoder model is also explored for tasks like anomaly detection and image denoising. It consists of:

- **Encoder**: Compresses the input data into a latent representation.
- **Decoder**: Reconstructs the original input from the compressed representation.

## Loss Functions

The models use a combination of loss functions tailored for medical image segmentation:

- **Dice Loss**: Handles class imbalance by focusing on the overlap between predicted and ground truth masks.
- **Cross Entropy Loss**: Evaluates pixel-wise classification performance.

## Evaluation Metrics

The performance of the models is assessed using metrics like:

- **Dice Coefficient**: Measures the overlap between predicted and true tumor regions.
- **Hausdorff Distance**: Evaluates the spatial accuracy of the segmentation.
- **Accuracy**: Indicates the overall correctness of the segmentation predictions.

## Real-World Applications

- **Accurate Diagnosis**: Helps in identifying and outlining tumor regions for better clinical decisions.
- **Treatment Planning**: Assists in precise targeting of tumor areas for surgery or radiation therapy.
- **Monitoring Disease Progression**: Enables consistent tracking of tumor changes over time.
- **Personalized Medicine**: Facilitates tailored treatment strategies based on accurate tumor characterization.

## Implementation

The models are implemented in Python using PyTorch and TensorFlow. Key steps include:

1. **Data Preprocessing**: Normalization, augmentation, and resizing of MRI scans.
2. **Model Training**: Multi-channel input with different MRI modalities, trained using stochastic gradient descent.
3. **Post-Processing**: Techniques like thresholding and morphological operations to refine the segmentation masks.
4. 

## Web-App Screenshots

![image](https://github.com/user-attachments/assets/fe6bfce3-e2fc-49e1-8882-775f5d34f701)
![image](https://github.com/user-attachments/assets/5ecce5f6-5473-4072-806f-db0e92e859c0)
![image](https://github.com/user-attachments/assets/db82f8da-d356-48be-b6a7-010f9330228a)

## References

- Ronneberger, O., et al., 2015. _U-Net: Convolutional Networks for Biomedical Image Segmentation_.
- Çiçek, Ö., et al., 2016. _3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation_.
- Sital, M., et al., 2020. _3D Autoencoders for Medical Image Reconstruction_.

## Conclusion

The project demonstrates the effectiveness of deep learning models in medical image segmentation, particularly for challenging tasks like brain tumor delineation. Future work includes experimenting with advanced architectures like Attention U-Net and integrating generative AI techniques for data augmentation.

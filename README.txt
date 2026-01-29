# Handwritten Digit Recognition using CNN

## Overview
This project implements a Handwritten Digit Recognition system using a Convolutional Neural Network (CNN) trained on the MNIST dataset.  
The application predicts handwritten digits using a trained deep learning model.

## Features
- CNN-based digit classification
- PyTorch implementation
- Streamlit web interface

## Project Structure

Hand-Written-Digit/
│
├── models/
│ └── mnist_cnn.pth
├── notebooks/
│ └── training.ipynb
├── src/
│ ├── model.py
| ├── app.py
├── requirements.txt
├── README.md
└── .gitignore

## How to Run

### Step 1: Install dependencies
```bash
pip install -r requirements.txt

## Run 
streamlit run app.py

## Model Details

CNN with convolution + pooling layers
Trained on MNIST dataset
Live prediction

# AIPI 590 (Applied Computer Vision) - Final Project
### Author: Rajiv Raman
### Institution: Duke University
### Date: April 30th, 2025

## Overview

The project task was to design a deep learning model that can automatically generate captions for images.

A live application has been fully deployed! You can use this to generate captions for your own images with the deep learning model: https://captioning-images.onrender.com/.

All data is sourced from the Flickr 8K dataset (contains images and five captions per image), which can be found here: https://www.kaggle.com/datasets/adityajn105/flickr8k. If you want to train the model locally, download the zip file from this website, then store the contents in a folder named "data" in your project directory. Once the model file is added to the project directory and all necessary requirements are downloaded, you can run **caption_eval.py** to assess the model's performance on the testing data, or you can run **app.py** in Streamlit to upload your own images for testing.

Though the model was trained and evaluated locally, it is also publicly deployed on Google Cloud here: https://console.cloud.google.com/storage/browser/imgcap-deep-model.

The full report on the project motivation and evaluation can be found here: https://docs.google.com/document/d/1_SWgdcM0hju0ZAICjY3w4xAzGxTDLB8mJuAAQkHYOQs/edit?usp=sharing.

# Arabic Character Recognition using Deep Learning

## Overview

- This project aims to recognize Arabic characters using deep learning techniques. The model is built and trained using Python in Jupyter Notebook, leveraging the 
  AHCD1 dataset from Kaggle.

## Libraries Used

- Matplotlib
- Pandas
- NumPy
- OpenCV (cv2)
- TensorFlow (tf)
- Keras (layers, models)
- Scikit-learn (train_test_split, confusion_matrix, classification_report)
- Seaborn (sns)
- ImageDataGenerator from TensorFlow.keras.preprocessing.image
- LabelEncoder from Scikit-learn
- os
- re
- to_categorical from TensorFlow.keras.utils

## Dataset

- The dataset used in this project is the [Arabic Handwritten Characters Dataset (AHCD1)](https://www.kaggle.com/datasets/mloey1/ahcd1) available on Kaggle. It contains handwritten characters in Arabic script, 
  which are used for training and testing the recognition model.

## Requirements

- Python 3
- Jupyter Notebook
- Libraries: TensorFlow, Keras, NumPy, Matplotlib, etc. (List dependencies)

## Usage

- Open the Jupyter Notebook file Arabic_Character_Recognition.ipynb.
- Follow the instructions within the notebook to:
- Load and preprocess the AHCD1 dataset.
- Build and train the deep learning model.
- Evaluate the model's performance.
- Test the model using the provided test code with an image.
- Or you can use the model that is already trained.

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgments
- [AHCD1](https://www.kaggle.com/datasets/mloey1/ahcd1) Dataset on Kaggle

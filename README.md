# AI-Based Waste to Recovery System

An AI-powered mini project that identifies waste items from images, recommends the best recovery route, and estimates the financial and environmental value of proper waste handling.

## Project overview

This application uses a deep learning model built with TensorFlow/Keras and MobileNetV2 to classify waste into categories such as cardboard, glass, metal, paper, plastic, organic, and trash. It then suggests the most suitable recovery method such as recycle, compost, reuse, energy recovery, or safe disposal.

It also computes:

- Recovery value in INR for the estimated waste weight
- CO2 saved in kilograms
- Environmental impact and sustainability score
- Historical tracking of all predictions in a CSV file

## Exact folder structure

```text
AI_Waste_Recovery_System/
├── app.py
├── class_names.json
├── confusion_matrix.png
├── dataset/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   ├── organic/
│   ├── trash/
│   └── ...
├── history.csv
├── recovery_engine.py
├── requirements.txt
├── README.md
├── train_model.py
├── training_history.png
├── waste_model.h5
└── sample_images/
    └── sample_waste.jpg  (add your own image for quick testing)
```

## Dataset

Use the TrashNet or Kaggle Garbage Classification dataset.

Expected dataset structure:

```text
dataset/
├── cardboard/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
├── glass/
├── metal/
├── paper/
├── plastic/
├── organic/
└── trash/
```

### Download options

1. TrashNet dataset (widely used for waste classification)
   - https://github.com/garythung/trashnet
2. Kaggle Garbage Classification dataset
   - https://www.kaggle.com/datasets/mostafashahbazi/garbage-classification

## Installation

Use Python 3.10 or 3.11 for this project. TensorFlow 2.15 is compatible with those versions.

```bash
python --version
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Model training

From the project root, run:

```bash
python train_model.py
```

This will:

- Load and split the dataset into train/validation/test sets
- Train a transfer-learning MobileNetV2 classifier
- Save `waste_model.h5`
- Save `class_names.json`
- Save training curves and confusion matrix as PNG files
- Print a classification report with precision, recall, and F1-score

## Running the web app

Start the Streamlit UI:

```bash
streamlit run app.py
```

Then open the local URL displayed in the terminal, for example:

```text
http://localhost:8501
```

## Sample test image

Place a test image in the project root or in `sample_images/` and use that file path when testing the app.

Example path:

```text
sample_images/sample_waste.jpg
```

To test in the app, upload the sample file from the browser or adjust the image path if used in scripts.

## How the app works

1. The user uploads or captures an image of waste.
2. The model predicts the material class.
3. The app recommends the proper recovery method.
4. The user enters the approximate weight in kilograms.
5. The app estimates the value recovered and CO2 saved.
6. All predictions are saved to `history.csv` for analytics.

## Tips to improve accuracy if the result is below 85%

- Increase the number of images in each class.
- Use balanced class distribution across categories.
- Add more lighting variation and background diversity.
- Apply stronger augmentation and more epochs.
- Fine-tune the last MobileNetV2 layers after the frozen base model.
- Use a larger or more diverse dataset for better generalization.
- Clean the dataset by removing poor-quality or mislabeled images.

## Viva questions with answers

### 1. What is the main objective of this project?
The main objective is to classify waste images automatically and recommend the most suitable recovery method to reduce landfill usage and increase sustainable reuse of materials.

### 2. Why is MobileNetV2 used in this project?
MobileNetV2 is lightweight, efficient, and works well for image classification tasks on limited hardware. It also provides strong transfer learning performance with fewer computational resources.

### 3. What is transfer learning in this context?
Transfer learning means using a pre-trained model on ImageNet and adapting it to the waste classification task. This reduces training time and often improves accuracy when the dataset is small.

### 4. Why is the recovery engine important?
The recovery engine converts the predicted class into actionable waste management guidance. It suggests the correct recovery method, process steps, and calculates financial and environmental value.

### 5. How does the project help sustainability?
The project supports proper segregation, recycling, and resource recovery, which reduces waste sent to landfills, saves energy, and lowers carbon emissions.

## Notes

- This project assumes the dataset is organized in folders by class name.
- If the model is not yet trained, run `python train_model.py` before launching the app.
- The app logs all predictions into `history.csv` and can be used for reporting and trend analysis.

## Run commands

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## License

This project is intended for educational and academic use.

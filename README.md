# CHD Prediction Pipeline

## Overview
This project implements a machine learning pipeline for predicting 10-year Coronary Heart Disease (CHD) risk using Google Vertex AI Pipelines.

See detailed EDA/model selection and result here: [Report](doc/Report.pdf)

## Pipeline Architecture
### Overview

### Model Pipeline Overview
![Model Pipeline Overview](doc/model%20pipeline%20overview.png)

### Inference Pipeline Overview
![Inference Pipeline Overview](doc/inference%20pipeline%20overview.png)

### Initial Training Pipeline
The pipeline consists of the following main components:

1. **Data Preprocessing**
   - Column dropping (removing PatientID and A1C)
   - Train-validation split
   - Missing value imputation
   - Outlier handling (Winsorization)
   - Log transformation
   - Box-Cox transformation
   - Income binning
   - Feature combination
   - Feature encoding
   - Feature scaling

2. **Model Training**
   - Voting Ensemble classifier combining:
     - Logistic Regression
     - Support Vector Machine
     - Balanced Random Forest
     - LightGBM
     - Gaussian Naive Bayes

### Inference Pipeline
A separate pipeline for making predictions on new data, which includes:
- All preprocessing steps from the training pipeline
- Model inference using the trained ensemble model
- Output generation with PatientID and CHD predictions

## Requirements
- Python 3.7+
- Google Cloud Platform account
- Required Python packages:
  - kfp
  - google-cloud-pipeline-components
  - pandas
  - scikit-learn
  - lightgbm
  - imbalanced-learn==0.9.0
  - gcsfs
  - fsspec

## Setup Instructions

1. **Google Cloud Setup**
   ```bash
   # Set your project ID and location
   project_id = 'your-project-id'
   location = 'us-central1'
   ```

2. **Environment Setup**
   ```bash
   pip install kfp google-cloud-pipeline-components gcsfs fsspec
   ```

3. **Pipeline Execution**
   - Training Pipeline:
     ```python
     pipeline_job = aiplatform.PipelineJob(
         display_name='chd-prediction-pipeline',
         template_path='chd_prediction_pipeline.json',
         pipeline_root='gs://your-bucket',
         parameter_values={
             'input_dataset_path': 'gs://your-bucket/input-data.csv',
         }
     )
     ```
   
   - Inference Pipeline:
     ```python
     pipeline_job = aiplatform.PipelineJob(
         display_name='chd-prediction-inference-pipeline',
         template_path='chd_prediction_inference_pipeline.json',
         pipeline_root='gs://your-bucket',
         parameter_values={
             'evaluation_dataset_path': 'gs://your-bucket/evaluation-data.csv'
         }
     )
     ```

## Input Data Format
The pipeline expects CSV files with the following features:
- demographic information (age, education)
- medical measurements (BMI, blood pressure, cholesterol)
- behavioral factors (smoking status)
- medical history (hypertension)

## Output
The pipeline produces:
- Trained model artifacts
- Preprocessing parameters
- Final predictions in CSV format with PatientID and predicted CHD risk

## Pipeline Components
Each component in the pipeline is implemented as a separate Python function decorated with @component, ensuring modularity and reusability.

## Performance Metrics
The model's performance is evaluated using:
- Accuracy
- F1 Score (weighted)


# Person Recognition Model Using Ground Reaction Force (GRF) Data

A machine learning model that identifies individuals based on gait patterns using force plate data, achieving **92% accuracy**. Designed for privacy-conscious home automation systems.

## Key Features
- **Non-invasive biometrics**: Uses GRF data instead of cameras/microphones.
- **Advanced feature engineering**: 38 time/frequency-domain features (peak force, FFT harmonics, unloading rates).
- **Optimized model selection**: Support Vector Classifier (SVC) with Recursive Feature Elimination (RFE) outperformed Logistic Regression, GMM, and baselines.
- **Biomechanics integration**: Features validated against clinical gait analysis standards.

## Repository Structure
wscode/
├── Data/ # Raw and processed datasets
├── Figure Folder/ # Visualizations (correlation matrices, accuracy plots)
│
├── featureextraction.py # Time/frequency-domain feature extraction
├── freqfeaturedocument.py # Frequency analysis (FFT, PSD)
├── stepseparation.py # Heel-strike/toe-off detection algorithm
├── modeling.py # ML model training (SVC, GMM, Logistic Regression)
├── crossvalidation.py # K-fold validation and performance metrics
├── graphcreation.py # Visualization tools
│
├── FullGaitFeatures.csv # Processed feature dataset
├── ModelPerformance.txt # Accuracy metrics across models
├── README.md # This file

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/reponame.git
2. Install dependencies:
    pip install numpy pandas scikit-learn matplotlib scipy

## Usage
1. Feature Extraction 
    python featureextraction.py \
    --input Data/raw_grf_samples/subject01.csv \
    --output Data/processed/subject01_features.csv
2. Model Training 
    python modeling.py \
    --features FullGaitFeatures.csv \
    --model SVC_RFE \
    --n_features 12  # Optimal feature count from RFE
3. Evalution
    python crossvalidation.py --model SVC_RFE --k 5

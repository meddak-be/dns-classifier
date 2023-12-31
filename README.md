# DNS Request Classifier: Bot or Human

## Overview

This machine learning project classifies DNS requests as originating from either bots or humans. By analyzing DNS requests captured through `tcpdump`, the project employs various machine learning algorithms to differentiate between bot-generated and human-generated requests.

## Repository Structure

- `eval.py`: Script for evaluating the trained model on a test dataset.
- `features_extractor.py`: Utility for extracting features from DNS request data.
- `train.py`: Main training script for building and training the machine learning model.

## Getting Started

### Prerequisites

- Python 3.x
- Libraries: pandas, joblib, scikit-learn, etc. (see `requirements.txt` for a complete list)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/meddak-be/data-analysis-p1.git
   ```
    
2. Install required Python packages:
    
   ```bash
   pip install -r requirements.txt
   ```
    

### Usage

#### Training the Model

1. Prepare your training data in the required format.
2. Run the training script:
   ```bash
   pip install -r requirements.txt
   ```    

#### Evaluating the Model

- After training, evaluate the model's performance by running:
  ```bash
  python eval.py
  ```

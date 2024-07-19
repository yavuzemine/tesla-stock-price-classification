# Tesla Stock Price Classification

## Overview
This repository contains Python scripts and Jupyter notebooks used to classify Tesla (TSLA) stock price movements based on various technical indicators and machine learning models.

## Project Structure
- **data/**
  - Contains the dataset `TSLA.csv` used in the analysis.
  
- **notebooks/**
  - Jupyter notebooks for data exploration, feature engineering, model selection, and evaluation.
  - `1.0-Data-Exploration.ipynb`: Initial exploration of the dataset.
  - `2.0-Feature-Engineering.ipynb`: Feature engineering including exponential moving averages, stochastic oscillator, and relative strength index.
  - `3.0-Model-Selection-Evaluation.ipynb`: Model selection using logistic regression with cross-validation.
  
- **src/**
  - Python script `classify_tesla_stocks.py` containing the `ClassifyTeslaStocks` class.
  
- **README.md**
  - This file providing an overview of the project, its structure, usage instructions, results, authorship, and licensing information.

## Requirements
- Python 3.x
- Libraries: pandas, numpy, scikit-learn

## Usage
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yavuzemine/tesla-stock-classification.git
   cd tesla-stock-classification
   
Install the required libraries:

pip install -r requirements.txt

Execute the classification process:

python src/classify_tesla_stocks.py

Results

The chosen model, logistic regression, achieved an ROC AUC score of approximately 0.95 on the test data.
Feature importance analysis identified key indicators influencing the classification.

Author : Emine Yavuz 
Contact: emineyavuzgsu@gmail.com
License : This project is licensed under the GNU General Public License v3.0. See the LICENSE file for details.

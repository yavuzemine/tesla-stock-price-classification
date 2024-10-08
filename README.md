# Tesla Stock Price Classification

## Overview
This repository contains Python script and Jupyter notebook used to classify the daily change of Tesla (TSLA) stock closing price with a threshold 5%.

## Project Structure
- **Dataset**
  - The dataset was obtained from https://www.kaggle.com/datasets/varpit94/tesla-stock-data-updated-till-28jun2021.
  
- **notebook**
  - Jupyter notebooks for data exploration, feature engineering, model selection, and evaluation.
  - `1.0-Data-Exploration.ipynb`: Initial exploration of the dataset.
  - `2.0-Feature-Engineering.ipynb`: Feature engineering including exponential moving averages, stochastic oscillator, and relative strength index.
  - `3.0-Model-Selection-Evaluation.ipynb`: Model selection using logistic regression with cross-validation.
  
- **script**
  - Python script `classify_tesla_stocks.py` containing the `ClassifyTeslaStocks` class.
  
- **README.md**
  - This file providing an overview of the project, its structure, usage instructions, results, authorship, and licensing information.

## Requirements
- Python 3.10
- Libraries: pandas, numpy, scikit-learn

## Usage
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yavuzemine/tesla-stock-classification.git
   
2. **Install the required libraries:**
   ```bash
   pip install -r requirements.txt

3. **Execute the classification process:**
   ```bash
   python src/classify_tesla_stocks.py

**Results**

The chosen model, logistic regression, achieved an ROC AUC score of approximately 0.95 on the test data.
Feature importance analysis identified key indicators influencing the classification.

Author : Emine Yavuz 

Contact: emineyavuzgsu@gmail.com

License : This project is licensed under the GNU General Public License v3.0. See the LICENSE file for details.

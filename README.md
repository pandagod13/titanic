# Titanic Survival Analysis Dashboard

This project analyzes the famous Titanic dataset to predict passenger survival using machine learning techniques. The interactive dashboard built with Streamlit allows users to explore the data, visualize key survival factors, and see model predictions.

## Live Demo
Streamlit App: [https://titanic-vuwkkhwfbmv4amu9yqtwgw.streamlit.app/](https://titanic-vuwkkhwfbmv4amu9yqtwgw.streamlit.app/)
Kaggle Dataset: [https://www.kaggle.com/competitions/titanic/data](https://www.kaggle.com/competitions/titanic/data)

## Features
- **Data Exploration**: View the Titanic dataset and understand its structure
- **Interactive Visualizations**:
  - Bar charts showing survival rates by various factors (Sex, Class, Age, etc.)
  - Combined feature analysis (Gender + Age Category)
- **Machine Learning Models**:
  - Logistic Regression with 83% accuracy
  - Decision Tree visualization showing key survival factors
- **Model Evaluation**:
  - Confusion Matrix
  - Classification Report with precision, recall, and F1 scores

## Key Findings
- **Gender**: Being female was the strongest predictor of survival
- **Class**: First-class passengers had significantly higher survival rates
- **Age**: Children (especially girls) had better chances of survival
- **Cabin**: Having cabin information correlates with higher survival rates
- **Family Size**: Medium-sized families (3-4 members) had better survival chances

## Technical Implementation
- Python with Pandas for data preprocessing
- Scikit-learn for machine learning models
- Matplotlib and Seaborn for visualizations
- Streamlit for interactive web dashboard

## Getting Started
1. Clone this repository
2. Install requirements: `pip install -r requirements.txt`
3. Run the Streamlit app: `streamlit run streamlit_app.py`

## Project Structure
- `streamlit_app.py`: Main application with interactive dashboard
- `Titanic.ipynb`: Jupyter notebook with exploratory data analysis
- `train.csv`: Training dataset from Kaggle

## Future Improvements
- Add feature importance visualization
- Implement model comparison (Random Forest, XGBoost, etc.)
- Add survival prediction for new passenger profiles
- Create more advanced visualizations (PCA, t-SNE) to show passenger groupings

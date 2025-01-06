# Housing Cost Predictor

## Project Overview, Objectives, and Goals
The Housing Cost Predictor project aims to predict the sale prices of houses based on various features such as the number of bedrooms, square footage, and location. Using a dataset from the Kaggle competition **"House Prices: Advanced Regression Techniques"**, the project involves several steps, including data cleaning, exploratory data analysis (EDA), feature engineering, and model training. 

The main objective was to build a machine learning model that accurately predicts house prices, providing valuable insights for both real estate professionals and individuals interested in the housing market.

### Key Goals:
- Clean and preprocess the data for modeling.
- Explore the relationships between features and target values.
- Train machine learning models and evaluate their performance.
- Select and pickle the best performing model for future use.

## Methodology

### Data Collection and Preprocessing
The dataset was sourced from Kaggle's [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) competition. The data was first cleaned to handle missing values, outliers, and categorical variables. Various preprocessing steps, including feature scaling and encoding, were performed to prepare the data for machine learning models.

### Model and Algorithms
A **Random Forest Regressor** was selected as the model due to its ability to handle complex relationships and its resilience to overfitting. The model was evaluated using performance metrics such as **Root Mean Squared Error (RMSE)** and **R² (Coefficient of Determination)**.

### Performance Metrics
- **RMSE**: 25,370.11
  - On average, the model's predictions deviate from the actual sale prices by about $25,370. Given that the standard deviation (std) of SalePrice is $74,202, this RMSE is relatively low, suggesting good predictive accuracy.
- **R²**: 0.87
  - This indicates that 87% of the variance in house prices is explained by the model, which is a strong result for a regression model.

### Visualizations
- Training/validation curves and confusion matrices (for categorical analysis) can be found in the notebooks and the saved outputs.

### Potential Next Steps
- Implement further hyperparameter tuning to improve the Random Forest model.
- Experiment with other regression models (e.g., Gradient Boosting, XGBoost).
- Deploy the model as a web application for real-time predictions.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Contributing](#contributing)
4. [License](#license)
5. [Credits and Acknowledgments](#credits-and-acknowledgments)

## Installation

To install the necessary dependencies, follow these steps:

1. Clone this repository:
    ```bash
    git clone https://github.com/markhugo/Housing-Cost-Predictor.git
    ```
2. Navigate into the project directory:
    ```bash
    cd house-price-prediction
    ```
3. Create a virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```
4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Load the dataset**:
    The dataset is loaded from the provided Kaggle competition link, or you can download it and place it in the `data/` folder.

2. **Run the Jupyter notebook**:
    Launch the Jupyter notebook to explore the data, preprocess it, and run the model:
    ```bash
    jupyter notebook
    ```

3. **Predict house prices**:
    After training, you can use the pickled model to predict house prices on new data:
    ```python
    import pickle
    model = pickle.load(open('best_model.pkl', 'rb'))
    prediction = model.predict(new_data)  # 'new_data' is a prepared data array
    ```
    Note: There are some configurations that are still being worked out such as applying the same preprocessing techniques to the new data.
          The pickled model expects the same dimensions for input data.

## Contributing

Contributions are welcome! If you’d like to contribute to this project, please follow the steps below:

1. Fork this repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a pull request.

Please ensure that your code follows the style guidelines and includes tests for new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Credits and Acknowledgments

- **Dataset**: Kaggle - [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- **Model**: Random Forest Regressor from scikit-learn
- **Libraries Used**: pandas, numpy, scikit-learn, matplotlib, seaborn


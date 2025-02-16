# Backpack-Price-Prediction

This project is part of the 2025 Kaggle Playground Series competition, where the goal is to predict the price of backpacks based on various attributes. The dataset includes categorical and numerical features such as brand, material, style, size, and weight capacity.

## Dataset

The dataset consists of the following features:

- Categorical Features: Brand, Material, Style, Color, Size

- Numerical Features: Compartments, Weight Capacity (kg)

- Target Variable: Price

## Model

The model is built using TensorFlow and Keras and follows these steps:

**1. Data Preprocessing**

  - Handling missing values with SimpleImputer.
  
  - Encoding categorical variables using Label Encoding.
  
  - Scaling numerical features using StandardScaler.
  
  - Splitting data into training and validation sets.

**2. Neural Network Architecture**

  - Embedding layer for categorical data representation.
  
  - Fully connected layers with ReLU activation.
  
  - Output layer with a single neuron for price prediction.

**3. Model Training**

  - Optimizer: Adam
  
  - Loss function: Mean Squared Error (MSE)
  
  - Metrics: Mean Absolute Error (MAE)
  
  - Number of epochs: 25
  
  - Batch size: 32

**4. Evaluation & Visualization**

  - Loss Curves: To track training and validation performance.
  
  - Histogram of Prices: Distribution of predicted vs. actual prices.
  
  - Scatter Plot: Comparison of actual vs. predicted values.

## Results

- The model was trained and evaluated, achieving a reasonable RMSE for backpack price prediction.

- Predictions were generated for the test dataset and saved in `submission.csv`.

## Submission Format

The final submission file follows this format:

```bash
id,Price
300000,81.411
300001,75.293
300002,90.567
```

## Running the Model

- Requirements
```bash
pip install numpy pandas tensorflow scikit-learn matplotlib seaborn
```

- Execution

Run the `Price_Prediction` script 

<em>This will preprocess the data, train the model, visualize the results, and generate a submission file.</em>

## Future Improvements

- Hyperparameter tuning to improve accuracy.

- Experimenting with different neural network architectures.

- Using advanced feature engineering techniques.

## Clone Repository

To download and work with this dataset, run:
```bash
git clone https://github.com/danaelshrbiny10/Backpack-Price-Prediction.git
```

## Dataset
You can use data from [kaggle](https://www.kaggle.com/competitions/playground-series-s5e2) 

## LICIENCE
This project is licensed under the terms described in the [licience](./LICENSE) File.
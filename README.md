# COVID-19 Decision Tree Classification

This project implements a decision tree model for classifying COVID-19 data. The model includes the calculation of various metrics such as entropy, information gain, and gain ratio. Special consideration is given to the "AGE" feature, as it is a regression attribute. The goal of the project is to predict the classification of COVID-19 cases based on various attributes, using a decision tree algorithm with a focus on accuracy.

## Key Features:

### 1. Data Preprocessing:
- Missing values in the `PREGNANT` column are replaced with a value of 2.
- Unwanted columns like `INTUBED` and `ICU` are dropped from the dataset.
- The target column `CLASIFFICATION_FINAL` is used to split the dataset into training and testing sets.

### 2. Metric Calculations:
- **Entropy**: Measures the impurity or uncertainty in a dataset.
- **Information Gain**: Determines the effectiveness of an attribute in classifying data.
- **Gain Ratio**: Adjusts information gain by normalizing it with the split information, improving decision tree performance in the presence of attributes with many distinct values.
- **Age-Specific Calculations**: A custom implementation is used to compute the best threshold for the `AGE` feature, which is treated as a regression attribute.

### 3. Decision Tree Construction:
- The decision tree is built recursively, selecting the best attribute based on the highest gain ratio at each step.
- The tree stops growing either when a maximum depth is reached or when the data within a node is homogeneous.

### 4. Prediction and Evaluation:
- The model uses a custom-built decision tree to predict the classification for the test data.
- The performance is evaluated using the **F1 Score**, which balances precision and recall, with an emphasis on the performance of the classifier on imbalanced datasets.

### 5. F1 Score: 
- The F1 score is used to evaluate the model's performance, especially when the dataset is imbalanced. In this case, it is calculated using the `f1_score` function from `sklearn.metrics`, with the `average='weighted'` parameter to handle class imbalance effectively.

## Code Overview:

The code includes several key components:

- **Preprocessing**: Cleans the dataset by handling missing values, removing unwanted columns, and converting the target variable into a binary classification.
- **Entropy & Gain Calculations**: Functions like `entropy()`, `information_gain()`, and `gain_ratio()` are used to calculate entropy and information gain, essential for the decision tree algorithm.
- **Age Thresholding**: Special handling for the `AGE` feature, including calculating the best threshold for splitting data based on age.
- **Decision Tree Construction**: A recursive function `build_tree()` is used to build the decision tree, where the root node is selected based on the highest gain ratio for each attribute.
- **Prediction**: The `predict()` function applies the constructed decision tree to predict classifications for new data.
- **Model Evaluation**: The model's performance is evaluated using the F1 score.

## Evaluation Result:

The F1 score is calculated for the decision tree's predictions on the test dataset. The value provides an indication of how well the model balances precision and recall. For this project, the calculated F1 score is:
0.79


This score suggests that the model is performing reasonably well, but there may be room for improvement, especially in handling class imbalances.

## Conclusion:

This project demonstrates the application of decision tree algorithms for classification tasks, with a particular focus on handling specific features like `AGE` (regression attribute). The F1 score indicates that the model is fairly effective, though further tuning, such as adjusting the maximum depth or experimenting with different splitting criteria, could improve performance.


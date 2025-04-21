# Price Range Classification with Machine Learning

This project uses machine learning techniques to predict the price ranges of mobile devices. It aims to classify devices into different price categories based on various device features.

## Project Structure

The project includes the following key steps:

### 1. Data Loading and Exploration
The dataset containing device features is loaded and explored. In this step, the basic structure of the dataset, missing data, and descriptive statistics are examined.

### 2. Data Cleaning and Preprocessing
Missing data is identified and appropriately filled. For numerical columns, the mean is used, and for categorical columns, the mode is used to fill missing values. Additionally, numerical features are scaled. Outliers in the dataset are also identified and removed.

### 3. Feature Engineering and Transformation
Certain features in the dataset are transformed into meaningful categories. For example, numerical values such as talk times are categorized into short, medium, and long durations. Additionally, categorical variables are transformed into numerical values.

### 4. Model Creation and Training
The dataset is split into training and testing sets. Machine learning models, such as the Random Forest Classifier, are trained to predict the price range of devices. The model is optimized to achieve high accuracy on the training set.

### 5. Model Evaluation
The model is evaluated using the test dataset. Performance metrics such as accuracy, precision, recall, etc., are calculated, and the model's performance is analyzed. Furthermore, feature importance is analyzed to understand which features contribute most to the model’s predictions.

### 6. Model Saving and Usage
The trained model is saved and stored in a file for future use in predictions.

## Technologies Used

- **Python**: Python was used for this project. It is a powerful language for data analysis and machine learning tasks.
- **Pandas**: Pandas was used for data analysis and manipulation. It provides essential functionality for loading datasets, cleaning data, and performing statistical analysis.
- **Scikit-learn**: Scikit-learn was used for creating, training, and evaluating machine learning models. It provides efficient tools for classification, regression, and model validation.
- **Joblib**: Joblib was used to save and load the trained model for future use.

## Project Flow

### 1. Data Loading and Exploration
The dataset is first loaded and the basic structure is checked. Missing values are detected, and data types for categorical and numerical features are identified.

### 2. Data Cleaning
Missing values are filled, and outliers or erroneous data are removed. The dataset is cleaned to improve its quality and make it suitable for model training.

### 3. Feature Transformation and Scaling
Numerical features are normalized, and categorical variables are transformed into numerical values. This ensures that the data is suitable for machine learning models.

### 4. Model Training
The dataset is split into training and testing sets. A classification model, such as Random Forest Classifier, is trained on the data. The model is fine-tuned to achieve optimal accuracy.

### 5. Model Evaluation
The trained model is tested on the test dataset. Performance metrics, including accuracy, precision, and recall, are calculated. A confusion matrix and classification report are generated to further assess the model’s performance.

### 6. Model Saving
After training, the model is saved for future use. It can be loaded at any time for making predictions on new data.

## Results

The model can predict the price range of mobile devices with high accuracy. The classification results are satisfactory, demonstrating that device price classification based on various features can be effectively achieved.

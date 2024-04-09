# Apple Quality Analysis and Prediction

## Overview
This project encompasses a Python script designed for analyzing and predicting the quality of apples using various attributes like size, sweetness, crunchiness, and ripeness. The script demonstrates a comprehensive approach to data handling, including cleaning, visualization, statistical testing, and predictive modeling using a Random Forest Classifier.

## Features
- **Data Cleaning**: Handles missing values and drops irrelevant columns to prepare the dataset for analysis.
- **Data Visualization**: Utilizes matplotlib and seaborn to create histograms and box plots for various apple attributes, providing an intuitive understanding of data distribution.
- **Statistical Testing**: Implements the Shapiro-Wilk test to assess the normality of the data distributions.
- **Correlation Analysis**: Computes and visualizes a correlation matrix to explore the relationships between different attributes.
- **Predictive Modeling**: Applies a Random Forest Classifier to predict the quality of apples based on their attributes, demonstrating both a full feature model and a reduced feature model focusing on non-invasive attributes.

## Prerequisites
To run this script, you will need:
- Python (version 3.6 or later)
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy

Ensure you have Python installed on your machine, and you can install all required Python packages using pip:
```
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## Understanding the Code

The script starts by importing necessary libraries and suppressing warnings for cleaner output.
It then loads the apple quality dataset from a CSV file, performs preliminary data exploration, and cleans the data by removing missing values and unnecessary columns.
Various plots are generated to visualize the attributes of the apples, providing insights into their distributions.
The Shapiro-Wilk test assesses the normality of the numeric columns.
A correlation matrix is computed and visualized to identify relationships between attributes.
The script concludes with the creation and evaluation of Random Forest Classifier models for predicting apple quality, first using all available features and then using a subset of non-invasive features.

## Contributing

Contributions to improve the script or extend its functionality are welcome. Please fork the repository, make your changes, and submit a pull request with a clear explanation of your modifications.

## License

This project is open-source and available under the MIT License.

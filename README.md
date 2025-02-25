# Housing Price Prediction - Machine Learning Project

## Overview

This project demonstrates a simple **Machine Learning model** to predict housing prices based on various features such as house age, distance to the nearest MRT station, latitude, longitude, and more. The model uses **Linear Regression** from Scikit-learn to make predictions.

## Technologies Used

- **Python 3.x**
- **Scikit-learn** (for machine learning algorithms)
- **Pandas** (for data manipulation)
- **Matplotlib** (for data visualization)

## Problem Statement

Given a dataset containing information about real estate properties, such as:
- Transaction date
- House age
- Distance to the nearest MRT station
- Number of convenience stores nearby
- Latitude and longitude

The objective of this project is to build a predictive model that estimates the **price of the house** per unit area.

## Dataset

The dataset used for this project is sourced from the UCI Machine Learning Repository:
- **Real Estate Valuation Data Set** (https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set)
- Features include transaction date, house age, proximity to MRT, number of convenience stores, and more.

## Features of the Dataset

- **X1 transaction date**: The date of transaction.
- **X2 house age**: Age of the house.
- **X3 distance to MRT station**: Distance to the nearest MRT station.
- **X4 number of convenience stores**: Number of convenience stores in the vicinity.
- **X5 latitude**: Latitude of the location.
- **X6 longitude**: Longitude of the location.
- **Y house price of unit area**: The target variable, which is the price per unit area of the house.

## Model Description

We use a **Linear Regression** model to predict the house price per unit area. The steps are as follows:
1. **Data Preprocessing**: The data is cleaned by handling missing values and performing basic data exploration.
2. **Model Training**: We split the data into training and testing sets, and train the Linear Regression model.
3. **Evaluation**: The model is evaluated using **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, and **R² Score**.

## Installation

Follow the steps below to run this project on your local machine:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/housing-price-prediction.git
   cd housing-price-prediction

---

### Key Sections Explained:
- **Project Overview**: Describes what the project does.
- **Technologies Used**: Lists the main libraries and tools.
- **Dataset**: Information about the dataset used in the project.
- **Model Description**: Explanation of the machine learning model used.
- **Installation**: Instructions on how to set up and run the project locally.
- **Usage**: How to execute the Python file and what to expect as output.
- **Evaluation Metrics**: Details about the metrics used to evaluate the model's performance.
- **Contributing**: Guidelines for contributing to the project if you'd like others to collaborate.
- **License**: You can include any licensing information here.

---

### How to Use:
1. Replace `https://github.com/yourusername/housing-price-prediction.git` with the actual link to your GitHub repository.
2. Customize or add more sections if needed (e.g., **Future Work**, **References**).

Let me know if you'd like to add or modify anything else in the README!

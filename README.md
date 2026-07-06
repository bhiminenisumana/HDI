#  🌎 Human Development Index (HDI) Prediction System
A Machine Learning-based web application that predicts Human Development Index scores using socio-economic indicators.

# Project Overview

The Human Development Index (HDI) is an important measure used to evaluate the overall development of a country. This project predicts the HDI score using Machine Learning based on key indicators such as Life Expectancy, Mean Years of Schooling, Expected Years of Schooling, and Gross National Income (GNI) per Capita. The application allows users to enter these values through a web interface and instantly receive the predicted HDI score.

---

# Project Objectives

- Develop a Machine Learning model to predict HDI scores.
- Analyze the relationship between socio-economic indicators and HDI.
- Build an interactive web application using Flask.
- Provide a simple and user-friendly interface for prediction.
- Demonstrate the practical use of Machine Learning in development analysis.

---

# Problem Statement

Human Development Index is one of the most important indicators used to measure a country's overall development. Calculating and analyzing HDI manually can be time-consuming and requires multiple socio-economic factors. This project aims to automate the prediction of HDI using Machine Learning, making the process faster, easier, and more accessible.

---

# Dataset Information

The project uses the Human Development Index dataset containing various socio-economic indicators collected from different countries.

### Input Features
- Life Expectancy
- Mean Years of Schooling
- Expected Years of Schooling
- Gross National Income (GNI) per Capita

### Target Variable
- Human Development Index (HDI)

The dataset was cleaned and prepared before training the machine learning model.

---

# Technology Stack

### Programming Language
- Python

### Machine Learning
- Scikit-learn
- Linear Regression

### Data Processing
- NumPy
- Pandas

### Data Visualization
- Matplotlib
- Seaborn

### Web Framework
- Flask

### Frontend
- HTML
- CSS
- JavaScript

### Development Tools
- Google Colab
- VS Code
- GitHub

---

# Project Workflow

1. Configure the project environment.
2. Import the required libraries.
3. Load and understand the dataset.
4. Perform data preprocessing.
5. Split the dataset into training and testing sets.
6. Train the Linear Regression model.
7. Evaluate the model performance.
8. Save the trained model using Pickle.
9. Develop the Flask web application.
10. Predict HDI through the web interface.

---

# Machine Learning Model Used

The project uses the **Linear Regression** algorithm.

Linear Regression learns the relationship between the selected socio-economic indicators and the Human Development Index score, allowing accurate prediction for new input values.

---

# Model Evaluation

After training, the model was evaluated using the testing dataset to verify its prediction performance.

The evaluation process ensures that the model provides reliable predictions for unseen data.

Evaluation techniques include:

- R² Score
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

---

# Model Performance

The trained model successfully predicts Human Development Index scores with good accuracy and consistent performance on the testing dataset.

The prediction results demonstrate that the selected socio-economic indicators have a strong relationship with the Human Development Index.

---

# Model Saving

The trained Linear Regression model is saved using the **Pickle** library.

Saved Model:

```
hdi_model.pkl
```

Saving the model allows it to be reused for prediction without retraining every time the application starts.

---

# Deployment

The prediction system is deployed as a Flask web application.

Users can:

- Enter the required socio-economic values.
- Click the **Predict HDI** button.
- View the predicted Human Development Index score.
- View the corresponding HDI category.

---

# Project Structure

```
HDI-Prediction/
│
├── dataset/
├── model/
│   └── hdi_model.pkl
├── notebooks/
├── screenshots/
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── app.py
└── README.md
```

---

# Key Features

- Predicts Human Development Index using Machine Learning.
- Interactive and responsive web interface.
- User-friendly input form with validation.
- Displays predicted HDI score instantly.
- Automatically classifies HDI into development categories.
- Simple Flask-based deployment.
- Easy to use and understand.

---

# Future Enhancements

- Add multiple Machine Learning algorithms for comparison.
- Improve prediction accuracy using advanced regression models.
- Deploy the application to cloud platforms.
- Add interactive charts and visual analytics.
- Enable batch prediction using CSV file uploads.
- Integrate real-time development datasets.

---

# Conclusion

The Human Development Index Prediction System demonstrates how Machine Learning can be used to estimate a country's HDI using important socio-economic indicators. By combining data preprocessing, model training, and a Flask-based web application, the project provides an efficient and user-friendly solution for HDI prediction. This project also serves as a practical example of applying Machine Learning techniques to solve real-world development analysis problems.

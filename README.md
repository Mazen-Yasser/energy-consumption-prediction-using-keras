# Energy Consumption Prediction using Keras

This project predicts energy consumption based on various factors like weather, time, and building characteristics. It uses a deep learning model built with Keras and TensorFlow to make accurate predictions.

---

## 📜 Description

The goal of this project is to build a model that can accurately forecast energy consumption. This is important for energy companies to manage supply and for consumers to optimize their energy usage. This project explores the entire machine learning pipeline, from data exploration and preprocessing to model building, evaluation, and deployment recommendations.

---

## 📊 Dataset

The dataset used in this project is `Energy_consumption_dataset.csv`. It contains 5000 entries and 12 columns.

### Features:

* **Month**: The month of the year (1-12).
* **Hour**: The hour of the day (0-23).
* **DayOfWeek**: The day of the week.
* **Holiday**: Whether the day is a holiday or not.
* **Temperature**: The temperature in degrees Celsius.
* **Humidity**: The relative humidity.
* **SquareFootage**: The square footage of the building.
* **Occupancy**: The number of occupants in the building.
* **HVACUsage**: Whether the HVAC system is on or off.
* **LightingUsage**: Whether the lights are on or off.
* **RenewableEnergy**: The amount of renewable energy generated.
* **EnergyConsumption**: The target variable, representing the total energy consumed.

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3 installed. You'll also need to install the required libraries.

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/Mazen-Yasser/energy-consumption-prediction-using-keras.git](https://github.com/Mazen-Yasser/energy-consumption-prediction-using-keras.git)
    ```
2.  Install the required libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```
    Create a `requirements.txt` file and add the following libraries:
    ```
    pandas
    numpy
    matplotlib
    seaborn
    plotly
    tensorflow
    keras
    scikit-learn
    xgboost
    lightgbm
    optuna
    ```

---

## Usage

To run the project, you can use the Jupyter Notebook `energy-consumption-prediction-using-keras.ipynb`. Make sure to place the `Energy_consumption_dataset.csv` file in the same directory as the notebook.

---

## 🤖 Model Training and Evaluation

This project explores and evaluates several machine learning models:

* **Keras/TensorFlow**: A deep learning model with multiple layers.
* **Random Forest Regressor**: An ensemble model from scikit-learn.
* **Gradient Boosting Regressor**: Another ensemble model from scikit-learn.
* **XGBoost**: A popular and powerful gradient boosting library.
* **LightGBM**: A fast and efficient gradient boosting library.

The models are evaluated using the following metrics:

* **Root Mean Squared Error (RMSE)**: Measures the average magnitude of the errors.
* **R-squared (R²)**: Represents the proportion of the variance for the dependent variable that's explained by the independent variables.

---

## 📈 Results

The results from the model evaluations show that the ensemble models, particularly the one combining the top 3 performing models, achieved the best performance with the lowest RMSE and highest R-squared values.

The best model achieved an RMSE of approximately **2.35** and an R-squared of **0.94**.

---

## 🔮 Future Work

Here are some potential next steps for this project:

* **Hyperparameter Optimization**: Implement hyperparameter tuning using tools like Optuna to further improve model performance.
* **Cross-Validation**: Use cross-validation for a more robust evaluation of the models.
* **Explore Time Series Models**: Investigate time series-specific models like LSTM and Prophet, which might capture temporal dependencies more effectively.
* **Data Augmentation**: Collect more diverse data to enhance the model's generalization capabilities.

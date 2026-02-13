# 📊 Data Generation using Modelling and Simulation for Machine Learning  
## Queue System Simulation using SimPy + ML Model Comparison

---

## 📌 1. Introduction

This project demonstrates how modelling and simulation can be used to generate synthetic datasets for Machine Learning applications.

We implemented a Discrete-Event Queue Simulation using the SimPy library and generated data by running 1000 simulations with randomly sampled parameters. The generated dataset was then used to train and compare multiple Machine Learning models.

The objectives of this project are:

- Generate data using simulation modelling  
- Study important system parameters and their bounds  
- Perform 1000 simulations  
- Train and compare multiple ML models  
- Identify the best performing model  

This project integrates Simulation + Data Science + Machine Learning.

---

## 📌 2. Simulation Model

### 🔹 Library Used
SimPy (Python-based Discrete Event Simulation library)

### 🔹 System Model
We simulated an M/M/c Queue System.

In this system:

- Arrivals follow exponential distribution (rate λ)
- Service times follow exponential distribution (rate μ)
- c represents number of parallel servers

This type of model is commonly used in:

- Server load modelling
- Cloud systems
- Network routers
- Bank/service counters

---

## 📌 3. Simulation Parameters

| Parameter | Description | Lower Bound | Upper Bound |
|------------|-------------|-------------|-------------|
| Arrival Rate (λ) | Customer arrival rate | 0.5 | 5 |
| Service Rate (μ) | Service processing rate | 1 | 6 |
| Number of Servers (c) | Parallel servers | 1 | 5 |

Each simulation was executed for a fixed simulation time of 1000 units.

---

## 📌 4. Dataset Generation

- Random parameters were generated within defined bounds.
- 1000 simulations were executed.
- For each simulation, the Average Waiting Time was recorded.

### 🔹 Final Dataset Features

- arrival_rate  
- service_rate  
- servers  
- avg_wait  

The generated dataset was saved as:

simulation_dataset.csv

---

## 📌 5. Machine Learning Models Used

The following regression models were trained and evaluated:

1. Linear Regression  
2. Ridge Regression  
3. Lasso Regression  
4. Decision Tree Regressor  
5. Random Forest Regressor  
6. Gradient Boosting Regressor  
7. K-Nearest Neighbors (KNN)  
8. Support Vector Regression (SVR)  
9. XGBoost  
10. MLP Neural Network  

---

## 📌 6. Evaluation Metrics

Models were evaluated using:

- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)  
- R² Score (Coefficient of Determination)  

---

## 📌 7. Results

The model comparison results were saved in:

model_comparison_results.csv

A performance visualization graph was generated and saved as:

model_comparison_plot.png

### 🔹 Observations

- Ensemble models such as Random Forest and Gradient Boosting performed better.
- Linear models struggled to capture non-linear queue behaviour.
- The best performing model achieved the highest R² score.

---

## 📌 8. Best Model

The best performing model was saved as:

best_model.pkl

This model can be loaded later for prediction without retraining.

---

## 📌 9. Project Structure

```
Queue-Simulation-ML/
│
├── simulation_dataset.csv
├── model_comparison_results.csv
├── model_comparison_plot.png
├── best_model.pkl
├── notebook.ipynb
└── README.md
```

---

## 📌 10. Conclusion

This project demonstrates how simulation modelling can be used to generate structured datasets for Machine Learning tasks.

Key Takeaways:

- Simulation is a powerful tool for synthetic data generation.
- Queue system parameters significantly affect system performance.
- Ensemble ML models perform better for non-linear systems.
- Simulation + ML integration is valuable in networking, cloud computing, and server optimization.

---

## 📌 11. Future Improvements

- Hyperparameter tuning using GridSearchCV  
- Cross-validation for improved reliability  
- Feature importance analysis  
- Real-world dataset comparison  
- Reinforcement learning-based optimization  

---

## 📌 Technologies Used

- Python  
- SimPy  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  
- XGBoost  

---

## 📌 Author

Pranshu Goel

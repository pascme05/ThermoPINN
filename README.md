# ThermoPINN

**Physics-Informed Neural Network for Motor Thermal Prediction**

## Overview

ThermoPINN combines classical RC thermal modeling with a physics-informed LSTM neural network to predict junction temperatures of electric motors. By integrating the underlying thermal physics into the loss function, the model achieves accurate temperature estimation while respecting physical laws.

Key features:
- RC thermal parameter identification from step response data
- Physics-informed loss for LSTM sequence prediction
- Data normalization and sliding-window sequence generation
- Flexible training, validation, and test split
- Optional visualization of predictions and errors

---

## Installation

Clone the repository and install required packages:

```bash
git clone https://github.com/yourusername/ThermoPINN.git
cd ThermoPINN
pip install -r requirements.txt
```

## Results

Table I: Results for RC, NN, and PINN model on generalisation set in therms of MAE/MSE/MAX error.

| Test ID   | RC                      | NN                     | PINN                  |
|-----------|-------------------------|------------------------|-----------------------|
| 60        | 9.83 / 162.56 / 36.42   | 2.57 / 10.47 / 16.85   | 1.77 / 6.34 / 12.34   | 
| 62        | 2.42 / 17.74  / 22.40   | 2.30 / 7.82 / 16.16    | 1.23 / 4.17 / 14.19   | 
| 74        | 9.91 / 181.12 / 40.43   | 2.10 / 6.66 / 8.13     | 1.35 / 2.72 / 6.97    |
| --------- | ----------------------- | ---------------------- | --------------------- |
| Avg       | -                       | -                      | 1.40 / 4.12 / 14.19   |

### NN-Results

| ID-60                                   | ID-62                                   | ID-74                                   |
|-----------------------------------------|-----------------------------------------|-----------------------------------------|
| ![ID_NN_ID60.png](docu/ID_NN_ID60.png)  | ![ID_NN_ID62.png](docu/ID_NN_ID62.png)  | ![ID_NN_ID74.png](docu/ID_NN_ID74.png)  |


### PINN-Results

| ID-60                                       | ID-62                                       | ID-74                                       |
|---------------------------------------------|---------------------------------------------|---------------------------------------------|
| ![ID_PINN_ID60.png](docu/ID_PINN_ID60.png)  | ![ID_PINN_ID62.png](docu/ID_PINN_ID62.png)  | ![ID_PINN_ID74.png](docu/ID_PINN_ID74.png)  |


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

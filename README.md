<!--────────────────────────────────────────────────────────────────────────────-->
  

---

## 🚀 Overview

Hi there! 👋  
This repo contains a **from-scratch** implementation of **Linear Regression** in Python—no ML libraries like scikit-learn used. We train on a synthetic Student Performance dataset (10,000 records) to predict a **Performance Index** (10–100).

---

## 📚 Table of Contents
- [🎯 Problem Statement](#-problem-statement)  
- [🗄️ About the Dataset](#️-about-the-dataset)  
- [🔢 Features & Target](#-features--target)  
- [📈 Linear Regression Theory](#-linear-regression-theory)  
- [🛠️ Tools & Tech Stack](#️-tools--tech-stack)  
- [🖼️ Results & Visuals](#️-results--visuals)  
- [⚙️ How to Run](#️-how-to-run)  
- [📫 Contact](#-contact)  

---

## 🎯 Problem Statement

Find the best-fit linear function that maps multiple input variables to a single output (Performance Index) using **Gradient Descent** and visualize model training & evaluation.

---

## 🗄️ About the Dataset

| Feature                             | Type     | Description                                        |
|-------------------------------------|----------|----------------------------------------------------|
| 🎓 Hours Studied                    | Numeric  | Average study hours in a day                     |
| 📝 Previous Scores                  | Numeric  | Scores obtained in prior tests                     |
| 💤 Sleep Hours                      | Numeric  | Average daily sleep hours                          |
| 📄 Sample Question Papers Practiced | Numeric  | Number of practice papers completed                |

**Target**  
| Performance Index | Numeric (10–100) | Rounded academic performance measure |

> **Note**: This is a **synthetic** dataset created for illustration. Relationships may not mirror real-world data.

---

## 🔢 Features & Target

1. **Hours Studied**  
2. **Previous Scores**  
3. **Sleep Hours**  
4. **Sample Papers Practiced**  

**→** **Target**: Performance Index

---
## 📈 Linear Regression Theory

Linear Regression models the relationship between inputs and a continuous output by fitting a linear function that minimizes prediction error.

---

### 1. Model Formulation

**Simple Linear Regression**  
![Simple Model](https://latex.codecogs.com/png.latex?\dpi{120}\bg{transparent}\color{white}\hat%7By%7D%20%3D%20b_1x%20%2B%20b_0)

**Multiple Linear Regression**  
![Multiple Model](https://latex.codecogs.com/png.latex?\dpi{120}\bg{transparent}\color{white}\hat%7BY%7D%20%3D%20\mathbf%7BX%7D\beta%20%2B%20b_0)

---

### 2. Cost Function (½MSE)

![Cost Function](https://latex.codecogs.com/png.latex?\dpi{120}\bg{transparent}\color{white}J%28\beta%2C%20b_0%29%20%3D%20\frac%7B1%7D%7B2m%7D\sum_%7Bi%3D1%7D%5Em%28\hat%7By%7D%5E%7B%28i%29%7D-y%5E%7B%28i%29%7D%29%5E2)

---

### 3. Gradient Descent Updates

![Update Rules](https://latex.codecogs.com/png.latex?\dpi{120}\bg{transparent}\color{white}\beta_j%20\leftarrow%20\beta_j%20-%20\alpha\frac{\partial%20J}{\partial\beta_j},\;b_0%20\leftarrow%20b_0%20-%20\alpha\frac{\partial%20J}{\partial%20b_0})

**Grad w.r.t. β**  
![Gradient Beta](https://latex.codecogs.com/png.latex?\dpi{120}\bg{transparent}\color{white}\frac{\partial%20J}{\partial\beta_j}%20=%20\frac{1}{m}\sum_{i=1}^m(\hat%7By%7D%5E%7B%28i%29%7D-y%5E%7B%28i%29%7D)x_j%5E%7B%28i%29%7D)

**Grad w.r.t. b₀**  
![Gradient b0](https://latex.codecogs.com/png.latex?\dpi{120}\bg{transparent}\color{white}\frac{\partial%20J}{\partial%20b_0}%20=%20\frac{1}{m}\sum_{i=1}^m(\hat%7By%7D%5E%7B%28i%29%7D-y%5E%7B%28i%29%7D))

---

### 4. Closed-Form Solution (Normal Equation)

![Normal Equation](https://latex.codecogs.com/png.latex?\dpi{120}\bg{transparent}\color{white}\beta%20=%20(X^TX)^{-1}X^TY,\;b_0%20=%20\bar{y}-\beta^T\bar{X})

---

## 🛠️ Tools & Tech Stack

- 🐍 **Python 3.8+**  
- 📊 **NumPy** – numerical computing  
- 📈 **Matplotlib** – plotting & visualization  
- 🐼 **pandas** – data handling  
- 🔧 **Git & GitHub** – version control  

---

## 🖼️ Results & Visuals

**Training Data Fit**  
![Training Plot](./images/Training%20Data%20Plot-SLR.png)  

**Testing Data Predictions**  
![Testing Plot](./images/Testing%20Data%20Plot-SLR.png)  

**Cost Convergence**  
![Cost vs Iterations](./images/Cost%20vs%20Iterations%20Plot-SLR.png)  

---

## ⚙️ How to Run

```bash
# Clone repo
git clone https://github.com/ApurvSardana/Linear-Regression.git
cd Linear-Regression

# Install dependencies 
pip install numpy pandas matplotlib

# Run the script
python MultipleLinearRegression/MultipleLinearRegression.py
```
---

## 📫 Contact

**Have questions or feedback?**

- 👤 **Name:** Himanshu Gupta  
- 📧 **Email:** [himanshugupta00235@gmail.com)  
- 🔗 **LinkedIn:** (www.linkedin.com/in/himanshu-gupta-383a6b220)  
- 💻 **GitHub:** (https://github.com/himanshugupta00235)

---


# THANKS! HAVE A NICE DAY😊

---


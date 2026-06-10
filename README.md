# 📊 Customer Churn Prediction

Predicting customer churn using CART decision trees, random forests, and multiple linear regression on a large-scale beginners dataset.

---

## Overview

In 2024, I participated in Carnegie Mellon University's four-week summer program intended to introduce students from around the country to Python, data science, and machine learning. Paired up with a professor, our team was tasked with parsing through this dataset of 500,000 customers and determining the top three factors that contributed to a customer churning their subscription. Defined by when customers stop doing business with a company, understanding and analyzing why customers churn is extremely important as businesses aim to increase their growth and profitability. The results slightly changed during my second attempt at this objective, but the procedures remained the same.

---

## Models & Results

Three models were trained and evaluated using Mean Squared Error (MSE) and Mean Absolute Error (MAE):

| Model | MSE | MAE |
|---|---|---|
| Random Forest | **0.0731** | **0.0731** |
| CART Decision Tree | 0.1001 | 0.1001 |
| Multiple Linear Regression | 0.1456 | 0.3170 |

Our **KPI** (Key Performance Indicator) was **Mean Absolute Error** (MAE) because we chose not to remove our outliers from the dataset before doing our analysis. MAE is less sensitive to outliers, meaning that the results of our algorithms were less swayed by having wider margins in terms of age and tenure.

**Random Forest was the best-performing model**, with the lowest error across both metrics.

<img src="images/churn_rf.png" width="600"/>

### Top Features (Random Forest)

1. Support Calls (Score > 0.20)
2. Total Spend
3. Age (In the original project, it was Payment Delay)

### Correlation Matrix

| Feature | Correlation |
|---|---|
| Support Calls | +0.5163 (strongest positive predictor) |
| Payment Delay | +0.3298 (second strongest positive) |
| Age | +0.1912 (third strongest positive |
| | |
| Total Spend | −0.3697 (strongest negative predictor) |
| Usage Frequency | -0.0533 (second strongest negative) |
| Tenure | -0.0213 (third strongest negative) |

<img src="images/churn_correlation.png" width="600"/>

---

## Key Takeaways

- Customers with **high support call volume** are the strongest churn signal — companies should invest in more customer service training and larger teams.
   - Nearly three out of five customers report that good customer service is vital for them to feel loyalty toward a brand *(Zendesk, 2018)*
   - 68% of customers say that they are willing to pay more for products and services from a brand known to offer good customer service experiences *(HubSpot, 2018)*
   - The data analysis showed that ⅓ of customers in this dataset churned as a result of having 5 or more support calls.
- Customers with **higher total spend** are actually *less* likely to churn, suggesting loyalty among higher-value customers.
   - That being said, customers will want to move their money away from the company if the service they receive is poor or unsatisfactory.
   - 65% of customers saying that they have changed to a different brand because of a poor experience *(Khoros, 2024)*
- **Random forests significantly outperformed** both the CART baseline and linear regression, reducing MSE by ~27% over CART.
   - Specifically for MLR, it's likely that the poor performance was due to it being less suitable for classification tasks such as this one.

---

## Recreating This Project

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Running the Notebook

```bash
git clone https://github.com/eliasangelss/customer-churn.git
cd customer-churn
jupyter notebook model.ipynb
```

---

## Project Structure

```
customer-churn/
├── model.ipynb       # Includes analysis, modeling, evaluation, and my notes
├── churn.csv         # Dataset; two file types, use as needed
└── README.md
```

---

## Dataset

- **Source:** [Kaggle Customer Churn Dataset](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset)
- **Size:** 505,206 entries
- **Target variable:** `Churn` (binary)

---


## Model Architecture

### *CART Model*
- **Max Leaf Nodes:** 5
- **Criterion:** Gini

### *Random Forest Model*
- **Number of Estimators:** 10
- **Criterion:** Gini

---

## Tools & Libraries

- Python, Jupyter Notebook
- scikit-learn (CART, Random Forest, Linear Regression)
- pandas, NumPy
- matplotlib, seaborn

---

## License

MIT

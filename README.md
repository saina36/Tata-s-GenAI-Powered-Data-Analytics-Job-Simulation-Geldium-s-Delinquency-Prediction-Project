# Tata-s-GenAI-Powered-Data-Analytics-Job-Simulation-Geldium-s-Delinquency-Prediction-Project
# 📘 **Geldium Delinquency Prediction – Machine Learning Project**

*A complete end-to-end credit-risk modeling pipeline*

---

## 📌 **Project Overview**

This repository contains a full predictive modeling workflow developed for **Geldium**, focusing on identifying customers who are likely to become **delinquent** (miss payments).

The project includes:

* Data preprocessing & feature engineering
* Statistical validation (Correlation, Coefficient Significance & VIF)
* Model building using Logistic Regression & Decision Tree
* Overfitting analysis
* Model selection logic
* Business interpretation of results
* Visualizations for insights
* Final deployment-ready model

This project was completed as part of the **Tata Data & AI Simulation (Frontier Talent Program)**.

---

# 🧠 **1. Business Problem**

Geldium wants to:

* Identify customers likely to default
* Engage them early with targeted interventions
* Reduce losses, improve repayment reliability
* Maintain fairness & regulatory compliance

The goal is to build a **simple, interpretable ML model** that helps the risk team detect delinquent customers early.

---

# 🧮 **2. Data Understanding**

The dataset contains:

* **Numeric variables**
  Income, Loan Balance, Credit Score, Utilization, Missed Payments, etc.

* **Categorical variables**
  Employment Status, Credit Card Type, Location

* **Monthly behavioral indicators**
  Month_1 – Month_6 (On-time, Late, Missed)

* **Target variable**
  `Delinquent_Account` (0 = no, 1 = yes)

---

# 🔧 **3. Feature Engineering**

Key engineered features:

| Feature              | Description                                       |
| -------------------- | ------------------------------------------------- |
| `Loan_to_Tenure`     | Loan balance divided by account tenure            |
| `Employment_Risk`    | Encoded employment stability (stable vs unstable) |
| `Hidden_Delinquency` | Sum of last 6 months’ Late/Missed signals         |
| `Miss_Rate`          | Proportion of missed payments                     |
| `Income_per_Debt`    | Income divided by total loan balance              |

**3 final selected features (used for model):**

```
Loan_to_Tenure  
Income  
Employment_Risk  
```

> *Optional feature `Hidden_Delinquency` was temporarily used for exploratory reproduction but excluded from the final model for fairness.*

---

# 📊 **4. Statistical Validation (Showcase)**

### ✔ Correlation matrix

Helps confirm that selected features are not strongly correlated with each other.

### ✔ VIF (Variance Inflation Factor)

Ensures no multicollinearity.
All selected features had **VIF < 2**, indicating independence.

### ✔ Coefficient significance (Logistic Regression)

Used to demonstrate interpretability and statistical relevance.

---

# 🤖 **5. Machine Learning Models Used**

Two models were trained for comparison:

### **🔹 Logistic Regression**

* Baseline interpretable model
* Class_weight = 'balanced' for imbalance handling
* Scaling applied (StandardScaler)

### **🔹 Decision Tree Classifier**

* Captures non-linear patterns
* max_depth = 4 for interpretability & controlled complexity
* Balanced class weighting

---

# 📈 **6. Model Evaluation Metrics**

Both models were evaluated using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**
* **ROC–AUC**
* **Overfitting check** (Train vs Test AUC difference)

---

# 🏆 **7. Final Model Selection**

After evaluation:

### ✔ **Decision Tree was selected as the final model**

Because:

* More stable generalization
* Captured non-linear interactions
* Provided business-friendly, rule-based explanations
* Lower overfitting than logistic regression

---

# 📉 **8. Overfitting Analysis**

| Model               | Train AUC | Test AUC | Overfit Gap                 |
| ------------------- | --------- | -------- | --------------------------- |
| Logistic Regression | Moderate  | Low      | High                        |
| Decision Tree       | Higher    | Higher   | **Lower gap → more stable** |

The Decision Tree demonstrated **better real-world reliability**.

---

# 🎯 **9. Key Insights**

* Customers with **high Loan-to-Tenure** ratios (large loans vs short tenure) have significantly higher delinquency risk.
* **Unstable employment** is a strong risk indicator.
* Customers with **lower income** are more likely to become delinquent.

These insights align with real-world lending behavior.

---

# 💾 **10. Saving and Exporting the Model**

The final model and scaler were exported using:

```python
import joblib
joblib.dump(tree, "final_model.joblib")
joblib.dump(scaler, "scaler.joblib")
```

These files can be used in any future application (Flask API, Streamlit dashboard, etc.)

---

# 📊 **11. Visualizations Included**

* Correlation Heatmap
* Income Distribution Plot
* Loan-to-Tenure Histogram
* Employment Risk vs Delinquency
* Boxplot: Income vs Delinquency
* Pairplot of all 3 selected features + target

These visuals help stakeholders understand the data and model behavior.

---

# 🛡 **12. Responsible AI & Compliance**

The model follows:

### ✔ Fairness

Avoids features that directly encode protected attributes.

### ✔ Explainability

Final model (Decision Tree) provides clear human-readable rules.

### ✔ Minimal leakage

Monthly delinquency features were removed to avoid unfair predictions.

### ✔ Compliance

Meets expectations under **IFRS 9**, **Basel II/III**, and internal audit requirements.

---

# 🚀 **13. Future Improvements**

* Use LightGBM or XGBoost with monotonic constraints
* Add calibration (Platt scaling)
* Deploy real-time scoring pipeline
* Build agentic AI collections strategy (Task 4)

---

# 🧩 **14. Task 4: AI-Powered Collections Strategy (Summary)**

The model feeds into a high-level automated system:

1. Risk detection →
2. Customer segmentation →
3. Personalized outreach →
4. Smart feedback learning

Includes human-in-the-loop, fairness guardrails, and measurable KPIs.

---

# 🏁 **Conclusion**

This project demonstrates a complete, explainable, and business-ready credit risk pipeline—from feature engineering to model selection to responsible deployment strategy.



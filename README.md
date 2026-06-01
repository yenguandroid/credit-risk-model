# Credit Scoring Business Understanding

## 1. How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well-documented model?

The Basel II Accord established international standards for banking supervision and risk management, requiring financial institutions to measure, monitor, and manage credit risk systematically. Under Basel II, banks must demonstrate that their risk assessment models are reliable, transparent, and supported by sound methodologies.

As a result, credit scoring models should be interpretable and well-documented for several reasons:

* **Regulatory Compliance:** Regulators require financial institutions to explain how lending decisions are made and how risk estimates are calculated.
* **Model Validation:** Independent validation teams must be able to understand, test, and verify model assumptions, inputs, and outputs.
* **Auditability:** Internal and external auditors need clear documentation of data sources, feature engineering steps, model development procedures, and performance metrics.
* **Risk Governance:** Decision-makers must understand the factors influencing credit decisions to ensure that lending practices align with organizational risk appetite.
* **Fairness and Accountability:** Transparent models help identify potential biases and support responsible lending practices.

Therefore, Basel II encourages the use of models whose predictions can be justified and documented, especially when those models directly affect lending decisions and regulatory capital calculations.

---

## 2. Without a direct "default" label, why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?

In many real-world credit datasets, a direct indicator of customer default may not be available. In such situations, a **proxy variable** must be created to represent credit risk. A proxy is an alternative measure that approximates the concept of default using available information.

For example, in the credit dataset used for this project, customer behavior metrics such as transaction frequency, account activity, or a derived risk indicator may be used to estimate the likelihood of default.

### Why a Proxy Variable is Necessary

* The target variable is required for supervised machine learning.
* Without a default label, the model cannot learn the relationship between customer characteristics and repayment behavior.
* A proxy allows model development when historical default outcomes are unavailable.

### Business Risks Introduced by Proxy Variables

#### 1. Label Misclassification Risk

The proxy may not accurately represent true default behavior. Customers classified as high risk by the proxy may not actually default, while some future defaulters may be labeled as low risk.

#### 2. Model Bias

The model learns patterns associated with the proxy rather than actual default events, potentially introducing systematic bias into lending decisions.

#### 3. Reduced Predictive Reliability

Since the target is only an approximation, model performance on the proxy may not translate into accurate prediction of real-world credit losses.

#### 4. Regulatory Concerns

Financial regulators may question whether the proxy adequately represents credit risk and whether decisions based on it are justified.

#### 5. Business Decision Risk

Poor proxy selection can lead to:

* Approving risky borrowers who later default.
* Rejecting creditworthy customers.
* Increased credit losses.
* Reduced customer acquisition and profitability.

For these reasons, proxy variables should be carefully designed, validated, documented, and continuously monitored to ensure they remain aligned with actual business outcomes.

---

## 3. What are the key trade-offs between a simple, interpretable model (e.g., Logistic Regression with WoE) and a high-performance model (e.g., Gradient Boosting) in a regulated financial context?

Credit risk modeling often involves balancing predictive performance against interpretability and regulatory requirements.

### Logistic Regression with Weight of Evidence (WoE)

#### Advantages

* Highly interpretable and transparent.
* Coefficients clearly indicate the direction and magnitude of risk factors.
* Easy to explain to regulators, auditors, and business stakeholders.
* Stable over time and less prone to overfitting.
* Widely accepted in traditional credit scorecard development.

#### Disadvantages

* Assumes primarily linear relationships between predictors and risk.
* May fail to capture complex interactions and nonlinear patterns.
* Often produces lower predictive accuracy than advanced machine learning models.

---

### Gradient Boosting Models (XGBoost, LightGBM, CatBoost)

#### Advantages

* Typically achieve higher predictive performance.
* Capture nonlinear relationships automatically.
* Handle complex feature interactions effectively.
* Often improve risk discrimination and ranking power.

#### Disadvantages

* Less transparent and more difficult to explain.
* Require additional explainability tools such as SHAP.
* More challenging to validate and audit.
* Greater regulatory scrutiny due to model complexity.
* Increased risk of overfitting if not properly controlled.

---

### Trade-Off Summary

| Factor                          | Logistic Regression + WoE | Gradient Boosting |
| ------------------------------- | ------------------------- | ----------------- |
| Interpretability                | High                      | Low to Medium     |
| Regulatory Acceptance           | High                      | Medium            |
| Ease of Validation              | High                      | Medium to Low     |
| Predictive Performance          | Moderate                  | High              |
| Transparency                    | High                      | Low               |
| Feature Engineering Requirement | High                      | Moderate          |
| Complexity                      | Low                       | High              |
| Auditability                    | High                      | Moderate          |

### Recommended Approach

In regulated financial environments, institutions often begin with interpretable scorecard-based models because they are easier to justify and govern. However, advanced machine learning models such as Gradient Boosting may be used when higher predictive accuracy is required, provided that strong model governance, documentation, validation procedures, and explainability techniques (e.g., SHAP analysis) are implemented.

The optimal choice depends on the organization's balance between regulatory requirements, business objectives, risk tolerance, and predictive performance needs.

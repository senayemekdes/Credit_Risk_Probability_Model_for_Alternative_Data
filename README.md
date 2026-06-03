# Credit Risk Probability Model for Alternative Data - Bati Bank × Xente

## Credit Scoring Business Understanding

### How does the Basel II Accord influence the need for interpretable and well-documented models?

The Basel II Capital Accord emphasizes **risk-sensitive capital requirements** through the Internal Ratings-Based (IRB) approach. Banks must accurately estimate **Probability of Default (PD)**, **Loss Given Default (LGD)**, and **Exposure at Default (EAD)**. 

This requires:
- **High interpretability** — so risk teams and regulators can understand why a customer received a specific risk score.
- **Comprehensive documentation** — covering data sources, feature engineering, model assumptions, validation, and monitoring processes.
- **Ongoing validation** — including back-testing and stability monitoring.

In this project, we prioritize interpretable techniques (e.g., Logistic Regression with WoE/IV) alongside more powerful models, while using MLflow for full traceability.

### Why is a proxy variable necessary, and what business risks does it introduce?

The Xente dataset does **not** contain an explicit `default` label. Therefore, we must engineer a **proxy target** (`is_high_risk`) using **RFM (Recency, Frequency, Monetary)** analysis and clustering.

**Business Risks**:
- The proxy may not perfectly represent actual default behavior → possible misclassification.
- Higher false positives → rejecting good customers (lost business).
- Higher false negatives → approving risky customers (increased Non-Performing Loans).
- Regulatory risk — Basel II expects strong justification and validation of any proxy.
- Concept drift over time as customer behavior changes.

We will mitigate these through thorough validation, monitoring, and business rules.

### Key Trade-offs: Interpretable vs High-Performance Models

| Aspect                    | Interpretable Model (Logistic Regression + WoE) | High-Performance Model (XGBoost / LightGBM) | Recommendation in Regulated Context |
|--------------------------|--------------------------------------------------|---------------------------------------------|-------------------------------------|
| Interpretability         | Excellent                                        | Moderate (needs SHAP)                       | Preferred                          |
| Predictive Performance   | Good                                             | Usually Superior                            | Use if validated                   |
| Regulatory Acceptance    | High                                             | Lower (extra explainability required)       | Interpretable first                |
| Implementation & Audit   | Easy                                             | Complex                                     | Interpretable                      |
| Overfitting Risk         | Lower                                            | Higher                                      | Interpretable                      |

**Our Approach**: We will build both and compare them, but prioritize interpretability for the final recommendation to leadership.

---
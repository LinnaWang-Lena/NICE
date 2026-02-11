# NICE: Neighborhood-Consistent Counterfactual Generation for Minority Class Augmentation

Class imbalance is recognized as one of the top challenges in data mining and remains a major factor degrading the performance of downstream models. While many data-driven approaches attempt to mitigate this issue by generating synthetic samples, they often introduce unrealistic or mislabeled instances near decision boundaries.

Counterfactual explanations provide a promising alternative by naturally generating near-boundary, label-flipping instances that help refine decision boundaries.

We propose **NICE**, a novel native instance-based counterfactual generation framework that constructs high-quality counterfactuals by adaptively combining existing instances with real feature values, rather than relying on interpolation.

NICE:
- identifies class-specific causal features  
- performs propensity score-based counterfactual matching  
- generates plausible minority-class instances  
- filters low-quality samples  

Experiments on 6 popular datasets demonstrate that NICE outperforms both widely used and recent counterfactual-based methods for addressing class imbalance.

---

## 🔧 Key Features

- Native instance-based counterfactual generation
- Causal feature-aware generation
- Propensity score-based matching
- Quality-controlled sample filtering

---

## 📊 Datasets

We evaluate NICE on 6 widely used datasets from the UCI Machine Learning Repository:

| Dataset | Samples | Features | Imbalance Ratio |
|--------|--------|---------|----------------|
| abalone_9V19 (ab_9V19) | 722 | 8 | 21.53% |
| abalone_13VR (ab_13VR) | 4178 | 8 | 19.58% |
| ecoli_3VR (ec_3VR) | 336 | 7 | 8.60% |
| glass_3VR (gl_3VR) | 214 | 9 | 11.59% |
| winquality-white_3-9V5 (wi_3-9V5) | 1482 | 11 | 58.28% |
| yeast_6VR (ye_6VR) | 1484 | 8 | 41.40% |

---

## 🧪 Experimental Setup

### Baselines
We compare NICE against 7 representative methods:

**Resampling methods**
- SMOTE
- ADASYN
- SL-SMOTE
- Borderline-SMOTE
- G-SMOTE

**Counterfactual-based methods**
- CFA
- DICE

### Classifiers
Experiments are conducted using 6 widely used classifiers:

- AdaBoost  
- Decision Tree  
- Logistic Regression (LR)  
- Random Forest (RF)  
- Support Vector Machine (SVM)  
- XGBoost  

---

## 📏 Evaluation Metrics

### Classification Performance
- **AUROC**
- **AUPRC**
- **F1 Score**

### Counterfactual Quality
- **Sparsity** — percentage of altered features  
- **Proximity** — distance between counterfactual and original instance  

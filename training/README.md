# Email Spam Classification – Training & Evaluation

This folder contains the complete machine learning workflow for training,
validating, and evaluating a BERT-based email spam classifier.

---

## Dataset

The dataset was created by merging **7 publicly available Kaggle email datasets**
to improve diversity, robustness, and generalization.

### Dataset Split
| Split | Samples |
|------|--------|
Train | 80,142 |
Validation | 10,017 |
Test | 10,018 |

Each sample contains:
- `message`: raw email text
- `label`: Spam / Not Spam

---

## Model

- **Architecture:** DistilBERT (`distilbert-base-uncased`)
- **Task:** Binary sequence classification
- **Reason for choice:**
  - High accuracy
  - Lower latency than full BERT
  - Suitable for production inference

---

## Training Configuration

- Max sequence length: 256
- Batch size: 16
- Epochs: 3
- Learning rate: 2e-5
- Optimizer: AdamW
- Loss: Cross-Entropy
- Mixed precision (FP16): Enabled when GPU available
- Best model selection: Based on **F1-score**

---

## Training Results

| Epoch | Train Loss | Val Loss | Accuracy | Precision | Recall | F1 |
|------|-----------|----------|---------|----------|--------|----|
1 | 0.0342 | 0.0473 | 0.9878 | 0.9922 | 0.9822 | 0.9872 |
2 | 0.0243 | 0.0328 | 0.9921 | 0.9892 | 0.9944 | 0.9918 |
3 | 0.0012 | 0.0416 | 0.9930 | 0.9927 | 0.9927 | 0.9927 |

The model shows stable convergence with no significant overfitting.

---

## Final Test Evaluation

Accuracy : 0.9929  
Precision: 0.9927  
Recall   : 0.9925  
F1-score : 0.9926  

### Confusion Matrix
```
[[5202   35]
 [  36 4745]]
```

- Balanced false positives and false negatives
- Strong generalization on unseen data

---

## Key Takeaways

- Large-scale dataset with proper train/val/test split
- Consistent metrics across validation and test sets
- High precision and recall, critical for spam detection
- Model is suitable for deployment in real-world systems

The trained model is exported and used directly by the


# IEEE Report Starter (Content Outline)

Use this as writing content, then place it into IEEE conference format.

## Abstract

Briefly summarize the goal: implementing a DNN image classifier from scratch with NumPy on the Digits dataset.
State key result (test accuracy), method, and why the dataset is suitable.

## Introduction and Background Work

- Motivation for image classification.
- Why DNNs are effective for nonlinear decision boundaries.
- Brief related work: classic MLPs and handwritten digit recognition.

## Theoretical and Conceptual Study

### Problem Setup

- Multi-class classification with 10 classes (digits 0-9).
- Input vector size: 64 (flattened 8x8 image).

### Model

- Architecture: Input -> Hidden ReLU layers -> Softmax output.
- Example architecture: 64 -> 64 -> 32 -> 10.

### Forward Pass

- Dense layer: `Z = A_prev W + b`
- ReLU: `A = max(0, Z)`
- Softmax for class probabilities.

### Loss

- Cross-entropy loss for multi-class classification.
- Optional L2 regularization term.

### Backpropagation

- Output gradient: `dZ_L = P - Y`
- Hidden gradients via chain rule and ReLU derivative.
- Parameter updates with mini-batch gradient descent.

## Experimental Setup

- Dataset: sklearn Digits.
- Train/test split (e.g., 80:20, stratified).
- Hyperparameters: learning rate, epochs, batch size, hidden layers, L2.
- Evaluation metrics: accuracy, loss, confusion matrix, classification report.

## Results and Analysis

Include:

- Table of experiments from `results/experiments_log.csv`.
- Accuracy/loss curves from `results/*_curves.png`.
- Confusion matrix discussion.

Discuss:

- Which hyperparameters improved test accuracy.
- Overfitting signs (train vs test gaps).
- Misclassified digits and likely causes.

## Conclusion and Future Work

Summarize what worked and final performance.
Possible future work:

- Add dropout.
- Add learning-rate scheduling.
- Compare with CNN baseline.
- Test on larger datasets (MNIST).

## References

Include sources such as:

- sklearn Digits dataset docs
- Standard neural network textbooks/papers
- Any figures or external resources used (with attribution)

# DL_Uncertainty_Quantification
A tensorflow implementation of established Bayesian and ensemble methods for uncertainty quantification of neural networks.

Methods:
 - Monte Carlo Dropout
 - Monte Carlo Dropout (Implemented on Last Layer only)
 - Ensemble (10 models trained with different initializations)
 - Stochastic Variational Inference
 - Stochastic Variational Inference (Implemented on Last Layer only)

Performance Metrics:
  Evaluates how efficiently the uncertainty estimates detect test-set prediction errors of the model
 - ROC (Receiver Operating Characteristics) - AUC
 - PR (Precision-Recall) - AUC

Model: LeNet

Datasets:
 - MNIST
 - K (kuzushiji) - MNIST

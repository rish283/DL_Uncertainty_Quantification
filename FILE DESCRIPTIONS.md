---
FILE DESCRIPTIONS
---


main_UQ.py: Main file.
 - Preprocesses Data
 - Trains model or calls trained model
 - Calls various UQ functions (UQ_methods.py) and performance quantification function (performance_metrics.py).

UQ_methods.py: Function containing implementaitons of the uncertainty quantification methods
INPUT: UQ method parameters.
OUTPUT: Uncertainty estimates corresponding to each test-set output of the model.

performance_metrics.py: Evaluates quality of uncertainty estimates
INPUT: Uncertainty estimates corresponding to each test-set output of the model.
OUTPUT: ROC-AUC, PR-AUC

FOLDERS:

Models:
 - lenet.py: contains model initialization function
 - Finetuned:
    - Contains trained models on MNIST/KMNIST: "md_...": main trained models, 
                                               "svi_...": model trained using stochastic variational inference, 
                                               "svi_ll_...": model trained using stochastic variational inference only on last layer
                                               
    - ensemble: Contains 10 different models for each dataset for ensemble method implementation

Data/Kuzushiji
 - Contains KMNIST train/test datasets

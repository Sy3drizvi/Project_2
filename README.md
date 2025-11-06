Both models were implemented using a consistent technical stack to ensure fair comparison:
Component Specification


Programming Language Python 3.10+
Deep Learning Framework TensorFlow 2.12.0 / Keras
Development Environment Jupyter Notebook / Spyder
Key Libraries NumPy, Pandas, Matplotlib, Scikit-learn
Image Processing Keras ImageDataGenerator
Hardware CPU: M4 Apple 

The implementation was done following software engineering best practi- ces with well-documented , detailed logging of experiments, and reproducibility. Both models were trained from scratch with random weight 
initialization to minimize the pretraining bias and fair comparison of the hyperparameter configurations.
Training time for the different models varied quite significantly; Model A took approximately 
90 minutes for 30 epochs while Model B was stopped after 45 minutes at the epoch
18 due to performance stagnation. This distinction demonstrates the computational efficiency
implications of hyperparameter choices.

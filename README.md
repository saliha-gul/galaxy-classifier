**Galaxy Classifier**

**Description**
The Galaxy Classifier is a deep learning model designed to classify galaxies based on their morphological features. This model is trained on the GalaxyMNIST dataset, which contains galaxy images categorized into distinct morphological classes. The goal of this project is to develop an intuitive classifier that can predict the morphological class of a galaxy given an image.

Frameworks Used:
- Keras (TensorFlow): The Keras implementation uses TensorFlow as the backend for training and model evaluation. (galaxy_mnist_keras.py)
- PyTorch: An alternative implementation is provided using the PyTorch framework. (galaxy_classification.py)

**Data Source**
The dataset used for training and evaluation in this project is derived from Galaxy Zoo, a citizen science project that classifies galaxies based on their shapes and other characteristics. Specifically, this project uses the Galaxy Zoo DECaLS data release, where labels for galaxy images are drawn.

The DECaLS survey data: Willett, K. W., et al. (2022). Galaxy Zoo: morphological classifications from citizen science. Monthly Notices of the Royal Astronomical Society, 509(4), 3966-3984. https://doi.org/10.1093/mnras/stac1179.

Additional Acknowledgements: I also acknowledge the DECaLS (Dark Energy Camera Legacy Survey) for providing the data used in this project. The DECaLS survey, which is part of the Dark Energy Survey, played a critical role in the collection of galaxy images.

**Requirements:**
Python 3.x
TensorFlow or PyTorch
NumPy
Matplotlib
pandas

there are two python code 

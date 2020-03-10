# Deep learning For Signal Processing

## [1.1 Image classification using linear model](https://github.com/dipam7/Deep_learning_for_signal_processing/blob/master/01_MNIST_linear.ipynb)

In this notebook, we perform image classification on the classic MNIST dataset. The model consists of 5 hidden layers, each with 1024 units, followed by an output layer. The output of every layer goes through a ReLU and has a dropout of 0.2

![test](https://github.com/dipam7/Deep_learning_for_signal_processing/blob/master/images/MNIST_model.png)

In just 20 epochs, our model achieves an accuracy of 97.68% on the test set. The loss function used is Cross Entropy loss along with Adam's optimizer. Learning rate is set to 0.001.

Once trained, we plot the best examples for every digit as detected by the model.

![test](https://github.com/dipam7/Deep_learning_for_signal_processing/blob/master/images/MNIST_best.png)

Finally we also use PCA and tSNE to reduce the dimensionlaity to 2 features and plot them. tSNE works better than PCA. We do this for the output of each layer of the network. 

**PCA**

![test](https://github.com/dipam7/Deep_learning_for_signal_processing/blob/master/images/dim_pca.png)

**tSNE**

![test](https://github.com/dipam7/Deep_learning_for_signal_processing/blob/master/images/dim_tsne.png)



## [1.2 Audio denoising using linear model](https://github.com/dipam7/Deep_learning_for_signal_processing/blob/master/02_linear_denoising.ipynb)

### Preprocessing
- Read audio files
- Take STFT on them
- Take absolute values
- Put them on the GPU

### Modeling
The model structure is shown below

![test](https://github.com/dipam7/Deep_learning_for_signal_processing/blob/master/images/linear_denoising_model.png)

### Loss function and Optimizer

We use Mean Squared Error between ground truth and prediction as the loss function. We use Adam's Optimizer with a learning rate of 0.001

### Training

We train the model for 3000 epochs with the final loss value near 0.0011

### Inferenece

Finally we take Inverse Fourier Transform to convert it back to audio files. We also calculate Signal to Noise ratio to check the quality of the output. Finally we also play the audio. A linear model is not really suited for this application but is a good starting point to understand the data and build more complex models.

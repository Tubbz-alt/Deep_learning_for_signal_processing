# Deep learning Systems

### 1.2 Audio denoising using linear model

**Preprocessing**
- Read audio files
- Take STFT on them
- Take absolute values
- Put them on the GPU

**Modeling**
The model structure is shown below

![test](link to image)

**Loss function and Optimizer**

We use Mean Squared Error between ground truth and prediction as the loss function. We use Adam's Optimizer with a learning rate of 0.001

**Training**

We train the model for 3000 epochs with the final loss value near 0.0011

**Inferenece**

Finally we take Inverse Fourier Transform to convert it back to audio files. We also calculate Signal to Noise ratio to check the quality of the output. Finally we also play the audio. A linear model is not really suited for this application but is a good starting point to understand the data and build more complex models.

# FacialKeypointsDetection
Combining my knowledge of computer vision techniques and deep learning architectures to build a facial keypoint detection system that takes in any image with faces, and predicts the location of 68 distinguishing key-points on each face

Defining the Convolutional Neural Network.

1. Defined a CNN with images as input and keypoints as output.
2. Constructed the transformed FaceKeypointsDataset.
3. Trained the CNN on the training data, tracking loss.
4. Saw how the trained model performs on test data.
5. Modified the CNN structure and model hyperparameters, so that it performs well *
* What does well mean?

"Well" means that the model's loss decreases during training and, when applied to test image data, the model produces keypoints that closely match the true keypoints of each face. 

Load, Visualize Data.

The facial keypoints dataset consists of 5770 color images extracted from the YouTube Faces Dataset.
1. 3462 of these images are training images, which are used to create a model to predict keypoints.
2. 2308 of these images are test images, which are used to test the accuracy of my model.
The information about the images and keypoints in this dataset are summarized in CSV files, which we can read in using pandas. The number of keypoints are 68, where each keypoint has (x,y) dimensions.

To prepare our data for training, I have used PyTorch's Dataset class.

Dataset class

torch.utils.data.Dataset is an abstract class representing a dataset. This class will allow us to load or shuffle batches of image/keypoint data using multiprocessing workers, and uniformly apply transformations to our data, such as rescaling and normalizing images for training a neural network.

A sample of our dataset will be a dictionary {'image': image, 'keypoints': key_pts}.

Defining a neural network in PyTorch and why use PyTorch?
1. PyTorch is definitely newer framework, but it's fast and intuitive whne compared to Tensorflow variables and sessions.
2. It is designed to look and act a lot like normal Python code: PyTorch neural nets have their layers and feedforward behavior defined in a class. Defining a network in a class means that you can instantiate multiple networks, dynamically change the structure of a modeland these class functions are called during training and testing.
3. It is also great for testing different model architectures. PyTorch networks are a modular, which makes it easy to change a single layer in a network or modify the loss function and see the effect on training.

PyTorch will be able to perform backpropagation by keeping track of the network's feedforward behavior and using autograd to calculate the update to the weights in the network.

Here are the steps that a training function performs as it iterates over the training dataset.

1. Prepares all input images and label data for training.
2. Passes the input through the network (forward pass)
3. Computes the loss (how far is the predicted classes are from the correct labels)
4. Propagates gradients back in to the network's parameters (backward pass)
5. Updates the weights (parameter update)

It repeats this process until the average loss has sufficiently decreased.


Overfitting
Convolutional, pooling, and fully-connected layers are all you need to construct a complete CNN, but there are additional layers that you can add to avoid overfitting, too. One of the most common layers to add to prevent overfitting is a dropout layer.

Dropout layers essentially turn off certain nodes in a layer with some probability, p. This ensures that all nodes get an equal chance to try and classify different images during training, and it reduces the likelihood that only a few, heavily-weighted nodes will dominate the process.
# FacialKeypointsDetection
Combining my knowledge of computer vision techniques and deep learning architectures to build a facial keypoint detection system that takes in any image with faces, and predicts the location of 68 distinguishing key-points on each face

Defining the Convolutional Neural Network.

1. Defined a CNN with images as input and keypoints as output.
2. Constructed the transformed FaceKeypointsDataset.
3. Trained the CNN on the training data, tracking loss.
4. Saw how the trained model performs on test data.
5. Modified the CNN structure and model hyperparameters, so that it performs well *
* What does well mean?

"Well" means that the model's loss decreases during training and, when applied to test image data, the model produces keypoints that closely match the true keypoints of each face. And you'll see examples of this later in the notebook.

Load and Visualize Data.

Training and Testing Data.

The facial keypoints dataset consists of 5770 color images extracted from the YouTube Faces Dataset.
1. 3462 of these images are training images, which are used to create a model to predict keypoints.
2. 2308 of these images are test images, which are used to test the accuracy of my model.
The number of keypoints are 68 where each keypoint has (x,y) dimensions.




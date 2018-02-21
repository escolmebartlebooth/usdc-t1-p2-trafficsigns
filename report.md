# **Traffic Sign Recognition**


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/reportvizualisation1.png "class distributions"
[image2]: ./examples/preproc1.png "base images"
[image3]: ./examples/preproc2.png "normalised"
[image4]: ./examples/preproc3.png "grayscaled"
[image5]: ./examples/preproc4.png "equalised"
[image6]: ./examples/preproc5.png "rotated"
[image7]: ./examples/AheadPrio.png "New Images"
[image8]: ./examples/viz1.png "Right of way"
[image9]: ./examples/viz2.png "Ahead Priority"
[image10]: ./examples/viz3.png "30 kph"
[image11]: ./examples/viz4.png "Traffic Free"
[image12]: ./examples/reportvizualisation2.png
[image13]: ./examples/viz5.png "Children Crossing"
[image14]: ./examples/featuremap.png "feature map"

---
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the shape property of the data arrays to access the statistics below:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

In the first vizualisation, the similarity of the label distribution between datasets can be seen and also that within a dataset, the classes are not evenly distributed.

![alt text][image1]

In the second vizualisation, it is clear that the distribution of classes within the training data is not uniform. This leads to the idea of whether to balance the distribution.

Balancing (by augmenting the dataset with more images from the classes with fewer frequencies) should mean that in a model that generalises well the model has not 'baked in' biases around image frequency.

![alt text][image12]

### Design and Test a Model Architecture

#### 1. Pre-processing of the data

The final pipeline for pre-processing was:

+ augment the training dataset to balance the class labels within the dataset
+ normalise the full colour image
+ convert to grayscale
+ apply and equalising operation to the image

Augmentation was achieved by iterating the dataset and adding rotated images for classes which were less frequent members of the dataset cf. the most frequent member.

The normalising was carried out so that the range of pixel values for the images were scaled between a fixed range of values. This is a necessary condition for making the optimisation faster and to avoid certain weights over-dominating or under-dominating the model.

The conversion to grayscale reduced the size of the data to be processed.

Each operation can be seen below from base image up to the final pre-processing step. Finally, the rotated image set shows how new images can be created from existing ones.

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]


#### 2. Describe what your final model architecture looks like.

My final model consisted of the following layers:

| Layer         	|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU		        |												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	| 1x1 stride, valid padding, outputs 10x10x16      									|
| RELU                  |                                       |
| Max pooling           | 2x2 stride, outputs 5x5x16 |
| Fully connected       | 400 to 120 |
| RELU                  |            |
| Drop Out              | 0.5        |
| Fully connected	| 120 to 84         									|
| RELU                  |         |
| Drop Out              | 0.5           |
| Fully connected       | 84 to 43   |
| Softmax	        | to calculate the probabilities of the predicted class        									|
|						|												|
|						|												|

#### 3. Describe how you trained your model.

To train the model, I prepared the augmented training and original validation data sets by passing them through the normalisation processes outlined above.

I then initialised the placeholder variables of the model (input data, weights, biases) and ran through for a number of epochs (default 100) mini-batches of a default size of 128. The model trained using a learning rate set initially to 0.001. Every 5 epochs i fed the validation set into the model to get a view of validation accuracy.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

I started with default hyper-parameters and no drop out (i.e. the LeNet model from the classroom). Using 100 epochs and a batch of 128 realised a validation score of 0.87.

Having set up the training pipeline (load data, augment, scale) i then tuned the model by adding drop out layers on each fully connected layer and then passing in combinations of parameters changing epochs, batch size, learning rate and drop out rate.

The best model found scored 0.961 on validation using 100 epochs ad batch size of 512, learning rate of 0.0001 and drop out of 0.5. In truth, the 2 key actions to improve on the base model were to augment the data set and to introduce drop out.

My final model results were:
* training set accuracy of 0.990
* validation set accuracy of 0.961
* test set accuracy of 0.928

The LeNet architecture was chosen as it was:
+ already programmed...
+ a well known image classifier

Other architectures i researched but did not implement were GoogLeNet and ResNet. I didn't implement them due to having achieved the validation accuracy of the project.

I also think more data would have helped the model generalise better as the higher trainig=ng accuracy compared to the testing accuracy suggested the model was over-fitting.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7]

Only the 4th image looked to be very difficult to classify due to the foliage obscuring the sign. Each of the others, to the human eye, looked to be good copies of existing classes in the data set. In particular:

The 1st 2 images have very simple structures and are distinct images in the set.

However, the 3rd and 4th inage should be harder to classify as:

The 3rd image would be harder to classify as it shares many facets with other images (the red circle with numbers inside)

The 5th image shares many facets with other images as well (red triangle with shapes inside)

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Right of Way      		| Right of Way		|
| Ahead Priority     			| Ahead Priority						|
| 30 kph					| Keep right								|
| Traffic Free	      		| Roundabout mandatory   	 				|
| Children Crossing			| Wild animals crossing						|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This is a lot less than the accuracy on the test set. Reasons may well be connected to the image rescaling process which may have over distorted the images and possibly the information contained in the background.  (I think prediction five is not far from the truth)

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction

The model is VERY certain of its predictions except for the 30 kph sign.

The predictions are set out below. I have not shown anything less than 0.1%

Image 1: Right of way

![alt text][image8]

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.963        			| Right of way 									|
| 0.032   				| Beware of Ice and Snow    					|
| 0.004					| Double Curve									|
| 0.0002      			| Priority Road         		 				|
| 0.0001			    | Pedestrians       							|

Image 2: Priority Ahead

![alt text][image9]

| Probability           |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| 0.999                 | Priority Road                                 |
| 0...                  | Roundabout Mandatory                          |
| 0...                  | end of no passing                             |
| 0...                  | Roundabout mandatory                          |
| 0...                  | Speed Limit 80 kph                            |

Image 3: 30 kph

![alt text][image10]

| Probability           |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| 0.719                 | Keep Right                                  |
| 0.130                 | Road Work                                      |
| 0.091                 | No Passing                                    |
| 0.024                 | Ahead Only                          |
| 0.015                 | Priority Road                         |

Image 4: Traffic Free

![alt text][image11]

| Probability           |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| 0.957                   | Roundabout mandatory                          |
| 0.011                   | Speed Limt 30 kph                             |
| 0.010                  | Speed Limt 20 kph                             |
| 0.006                  | Speed Limt 70 kph                          |
| 0.005                  | End all speed and passing limits              |

Image 5: Children Crossing

![alt text][image13]

| Probability           |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| 0.990                   | wild animals crossing                       |
| 0.0055                   | Roundabout mandatory                       |
| 0.0049                  | Speed Limit 120kph                          |
| 0....                  | Double Curve                                 |
| 0....                  | Speed Limit 100kph                         |


### Visualizing the Neural Network

#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text][image14]

The first convolutional network output looks to be picking up basic structure of the image with some features being the raw outline of the sign and others being what is contained within the sign.

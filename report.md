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

[image1]: ./examples/reportvisualization1.png "class distributions"
[image2]: ./examples/preproc1.png "base images"
[image3]: ./examples/preproc2.png "normalised"
[image4]: ./examples/preproc3.png "grayscaled"
[image5]: ./examples/preproc4.png "equalised"
[image6]: ./examples/preproc5.png "rotated"
[image7]: ./data/placeholder.png "Traffic Sign 1"
[image8]: ./data/placeholder.png "Traffic Sign 2"
[image9]: ./data/placeholder.png "Traffic Sign 3"
[image10]: ./data/placeholder.png "Traffic Sign 4"
[image11]: ./data/placeholder.png "Traffic Sign 5"
[image12]: ./examples/reportvizualisation2.png

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

In the second vizualisation, it is clear that the distribution of classes within the training data is not uniform. This leads to the idea of balancing the distribution
so that individual classes are not modelled to be less probable than others.

![alt text][image12]

### Design and Test a Model Architecture

#### 1. Pre-processing of the data

The final pipeline for pre-processing was:

+ augment the training dataset to balance the class labels within the dataset
+ normalise the full colour image
+ convert to grayscale
+ apply and equalising operation to the image

Augmentation was achieved by iterating the dataset and adding rotated images for classes which were less frequent members of the dataset cf. the most frequent member.
By adding more data and by balancing the class distribution i hoped to make the model more robust.

The normalising and equalising operations were invoked to help the model with feature extraction.

The conversion to grayscale reduced the size of the data to be processed without losing much information.

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
| Drop Out              | 0.5        |
| RELU                  |            |
| Fully connected       | 84 to 43   |
| Softmax	        | to calculate the probabilities of the predicted class        									|
|						|												|
|						|												|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of 0.949 
* test set accuracy of 0.926

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



## Udacity Self-Driving Cars: Project 2: Build a Traffic Sign Recognition Program

Overview
---
The goals / steps of this project are the following:
* Load the German Traffic Sign data set
* Explore, summarize and visualize the data set
* Design, train and test a Convolutional Neural Network model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the pickled data set from: https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip
2. Clone the project from here
```
git clone https://github.com/escolmebartlebooth/usdc-t1-p2-trafficsigns.git
cd usdc-t1-p2-trafficsigns
jupyter notebook Traffic_Sign_Classifier.ipynb
```

### Implementation Notes

#### Section 1: Data Loading, Vizualisation and Pre-processing

The basic premise is:
+ Load training, validation, testing datasets
+ show how classes are distributed within and across datasets
+ create a pipeline to augment training data to balance classes and increase data size
+ create a further pipeline to transform data to improve feature extraction

#### Section 2: Model Creation and Training

In this section, the model architecture is defined, training is carried out using mini-batch.

It is possible to train using different hyper-parameters such as batch size, training epochs, learning rate and so forth.

The model is validated on the validation set.

The current model achieves 0.961 on validation.

#### Section 3: Testing and Network Vizualisation

The final section shows the test set results (0.928) and prediction results with probabilities for 5 new images.

The model does not perform brilliantly, which i think is related to overfitting, in turn caused by not having enough data and still uneven class distributions.

The network vizualisation shows the output from the first convolution and indicates what each channel is learning about the images - i.e. which features are being extracted - which in this case looks to be the major shapes of the images (triangle, circle, etc)
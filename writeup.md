#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/random_input_image.png "Visualization"
[image2]: ./examples/data_distribution.png "Data Distibution"
[image3]: ./examples/distribution_augmented.png "Augmented Data Distribution"
[image4]: ./examples/grey_scale.png "Preprocessed Image"
[image5]: ./german-signs/1-30limit.png "Traffic Sign 1"
[image6]: ./german-signs/8-120limit.jpg "Traffic Sign 2"
[image7]: ./german-signs/14-Stop.png "Traffic Sign 3"
[image8]: ./german-signs/23-SlipperyRoad.jpg "Traffic Sign 4"
[image9]: ./german-signs/27-Pedestrians.png "Traffic Sign 5"
[image10]: ./examples/Softmax_1.png "Top 5 Softmax"
[image11]: ./examples/Softmax_2.png "Top 5 Softmax continued"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup

You're reading it! and here is a link to my [project code](https://github.com/dhirendraism/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Checked the input data.
Code in 2nd cell.
> 	Number of training examples = 34799
> 
> 	Number of testing examples = 12630
> 
> 	Image data shape = (32, 32, 3)
> 
> 	Number of classes = 43
> 
> 	Signs: 
> 
> 	0 Speed limit (20km/h)
> 
> 	...

####2. Randomly took an image and checked in visually.
Code in 3rd cell.

![alt text][image1]
####3. Checked the distribution of data.
Code in 4th cell.

![alt text][image2]
####4. Augmented the data.
Code in 8th & 9th cell.

* Took a random set of distortion operations.
* For each image picked a random index.
* Picked a random distortion.
* Applied the distortion in the image.
 
##### Distribution after augmentation.

![alt text][image3]

###Design and Test a Model Architecture

####1. Pre-Processing the imgaes.
Code in 11th cell.

* Transformed images to Grey Scale.
* Normalized images to handle outliers for brightness.

> Shape before preprocessing:  (34810, 32, 32, 3)
> 
> Shape after preprocessing:  (34810, 32, 32, 1)

####2. Validate processed image.
Code in 12th cell.

![alt text][image4]

####3. Model Architecture.
Code in 14th cell.

* I started off with the LeNet model and did some experiments by replacing the Max Pool by Average Pool in Lenet. This helped me in better extraction of features and improvement in accuracy by 2-4%. 
* Then changed the Input & Output filter depth which brought in some improvements but with increase in Epochs nothing significant was seen.
* Then experimented with the learning rate. by decreasing the rate the accuracy did increase but nothing significant either.
* Then with some help from internet I started changing the model to add more layers.
* Added another conv after layer one. This improved the accuracy in 20 Epochs by 5-6%.
* Did some experiment with Average and Maxpool again. Average pool gave in good results.
* Added a drop out layer just before last fully connected layer. This improved the accuracy further.
* Finally froze the model at this arch and started experimenting with depth params. Final values can be seen in 14th Cell.

Overall Architecture is:

* Convolution with Input = 32x32x3. Output = 30x30x6.
* Convolution with Input = 30x30x3. Output = 28x28x12.
* Average pooling with stride of 2,2. Input = 28x28x12. Output = 14x14x12.
* Convolution with Input = 14x14x12. Output = 10x10x32.
* Average Pooling with stride of 2,2. Input = 10x10x32. Output = 5x5x32.
* Flatten to 1x800.
* Fully connected network. 1x800 Input, 800x240 Weight, 1x240 Bias & Output = 1x240.
* Relu activation.
* Fully connected network. 1x240 Input, 240x168 Weight, 1x168 Bias & Output = 1x168
* Relu activation.
* Fully connected network. 1x168 Input, 168x43 Weight, 1x43 Bias & Output = 1x43.

| Layer         		|     Description| 
|:-----------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image| 
| Convolution 3x3   | 1x1 stride, Valid padding, outputs 30x30x6|
| RELU					|	Relu Activation|
| Convolution 3x3   | 1x1 stride, Valid padding, outputs 28x28x12|
| RELU					|	Relu Activation|
| Average pooling	| 2x2 stride,  outputs 14x14x12|
| Convolution 5x5	| 1x1 stride, Valid padding, outputs 10x10x32|
| RELU					|	Relu Activation|
| Average pooling	| 2x2 stride,  outputs 5x5x32|
| Flatten				| 1x800
| Fully connected	| 1x800 Input, 800x240 Weights, 1x240 Bias|
| RELU					| Relu Activation|
| Fully connected	| 1x240 Input, 240x168 Weights, 1x168 Bias|
| RELU					| Relu Activation|
| Dropout				| Probability 0.5|
| Fully connected	| 1x168 Input, 168x43 Weights, 1x43 Bias|

 

Learning Rate: 0.001
Sigma: 0.1
Batch Size: 128

Used AdamOptimizer for optimizing the learning.

####4. Evaluation Pipeline
Code in 20th cell.

I created a Evaluation Pipeline where Match_Percentage, Precision of prediction, Recall and Total accuracy was calculated. This helped me in identifying the Models accuracy and evaluate if Model for Underfitting or Overfitting.
####5. Training Model
Code in 21th cell.

While training the kept track of validation accuracy to see how does model behaves with Epochs increasing value. This helped me in finding the sweet spot for Epoch and Learning rate. Persisted the model on the drive for later retrieval for prediction and evaluations.
####6. Compared the accuracy for Training and Validation.
Code in 23rd cell.

This gave me an idea of overfitting and underfitting.

> Training Accuracy = 0.997

> Validation Accuracy = 0.945

###Test a Model on New Images

####1. Accuracy in test images.
Code in 24th cell.

> Test Accuracy = 0.929

####2. German Traffic Sign Images from web.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

The first image might be difficult to classify because ...

####3. Output
Code in 
Here are the results of the prediction:

| Image			  		|     Prediction| 
|:--------------------:|:---------------------------------------------:| 
| 30 km/h	      			| 30 km/h| 
| 120 km/h   				| 120 km/h|
| Stop						| Stop|
| Slippery Road	      	| Slippery Road|
| Pedestrians				| Pedestrians|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of ...

####3. Softmax for prediction of images.

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         		|     Prediction| 
|:------------------------:|:---------------------------------------------:| 
| .99         				| 30 km/h| 
| 1.0    						| 120 km/h|
| .97							| Stop|
| .76	      					| Slippery Road|
| .99				    		| Pedestrians|

##### Top 5 Softmax

![alt text][image10]
![alt text][image11]
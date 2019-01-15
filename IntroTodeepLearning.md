
# Introduction to Deep Learning
Deep learning is a deep topic (pun intended). It also happens to have a large vocabulary so don't fret if you feel overwhelmed: keep calm and ask google. There are links provided throughout this document as a resource for you to read into the topics (it's recommended to at least skim some but not necessary). Let's dive in. 

Note: There's quite a large amount of information here. Consuming it all in one sitting will be difficult.

## Table of Contents
* [Overview](#Overview)
* [Basic Neural Network](#Basic-Neural-Network)
* [Activation Functions](#Activation-Functions)
* [Layers](#Layers)
* [Optimization](#Optimization)
* [Generalizing Models](#Generalizing-Models)
* [Dealing With Data](#Dealing-with-Data)
* [Resources](#Resources)

## Overview
#### What is machine learning?
The ability for AI systems to acquire their own knowledge, by extracting patterns from raw data.

Machine learning is built on top of linear algebra, probability, statistics, and calculus.

#### Why is machine learning so popular now?
Many reasons. Some machine learning algorithms have been around for quite a while (perceptron was implemented in hardware in 1957) but there was a shortage of 1. computational power and 2. data to train the models with. 

The advent of the internet, portable devices, and other technologies has provided us with massive amounts of data while advances in hardware (GPUs - we can thank the video game industry for this one) and distributed systems have expanded our computational capabilities to the point where we can train larger networks.

Deep Learning itself has been around for quite a while. Known as Cybernetics in the 1940s-1960s, Connectionism in the 1980s-1990s, and finally as Deep Learning in its resurgence beginning in 2006.

#### What is the difference between machine learning and deep learning?
Deep learning is a subset of machine learning that is focused primarily on "deep" networks. They are called "deep" due to the large amount of hidden layers. See the image below.

![example nn](https://cdn-images-1.medium.com/max/1600/1*r0fxAZRpRGapPnC4bniDiQ.png)

###### From Deep Learning page 9
![venn diagram](https://i.gyazo.com/684c42227e3028edcd062fc9bdc43cdf.png)

###### From Deep Learning page 10
![flowchart](https://i.gyazo.com/37066206f801c66cf21f5afb9673f7bd.png)

#### Why deep learning?
Deep learning was created as a way for AI systems to handle intuitive problems such as identifying objects within a picture or recognizing spoken words. For comparison: humans can tell the difference between dogs and cats at a young age with little data.

Deep learning has also become more useful as more data has become available. Though there are instances of deep networks being trained successfully on very little data.

Deep learning creates its own representations of the data and this has been shown to outperform hand-designed representations.

![googledeep](https://image.slidesharecdn.com/jeffdean-170115012221/95/jeff-dean-at-ai-frontiers-trends-and-developments-in-deep-learning-research-8-638.jpg?cb=1484674187)

#### What problems can deep learning solve?
Theoretically, any. See [Universal Approximation Theorem](https://towardsdatascience.com/can-neural-networks-really-learn-any-function-65e106617fc6). Deep learning currently handles problems such as natural language processing and image recognition well.

## Basic Neural Network
Some useful links if you don't want to read this section:

* [What is a neural network](https://www.youtube.com/watch?v=aircAruvnKk) (video)
* [Quick intro to neural networks](https://ujjwalkarn.me/2016/08/09/quick-intro-neural-networks) (blog post)

![simplerNN](https://i.gyazo.com/511d2bbf4aee74c1be4afcd4ed847ea7.png)

Above is a simple Neural Network. There are two input, two hidden, and one output node. Information flows from the input nodes to the hidden nodes and finally to the output node. w stands for weight: they represent how much of an effect an input has on the next node. 

The input to node h1 is the sum of x1 x w11 and x2 x w21. The input to node h2 is the sum of x1 x w12 and x2 x w22.

This allows the network to learn *representations* or *features* inherent in the data. Note that this means that biases can also be learned in the data, which is why good and well-formed data is integral to a good model. The learning algorithms are only as good as the data we provide it.

The input to the output node is the sum of h1 x w3 and h2 x w4.

This allows the network to combine its knowledge of features to decide on an output. In deeper networks, or networks with more layers, the algorithm can learn simple representations and along each layer combine them into more complex representations. An example in image recognition is the network can recognize lines, which combine into shapes, which combine into small objects, which finally combine into the main subject of the image.


Also notice that the operations from input layer to hidden layer resemble that of a matrix vector multiplication.

![matrixweights](https://i.gyazo.com/5e20fe51922a5a29dfcc70581ef3706e.png)

Linear algebra is quite important for understanding machine learning. You can see here that the weights can be represented as a matrix. As the network increases in complexity, the algorithms will be dealing with higher dimension matrices aka *tensors*. For now lets think of each weight as a *parameter*.

![simpleNN](https://dzone.com/storage/temp/7913025-neural-network.png)

This above image introduces some new elements from the previous image. 

There is now an extra node in each layer called a *bias node* that will always output 1. The weights for the bias nodes determine how strong the bias is. This provides flexibility to the network by providing a value that is independent of the input. For example, let's assume all nodes can have some value between 0 and 1. If the output of h1 and h2 were 0, the network's output would always be 0 in this case since any weight multiplied by 0 is 0. By introducing the bias node, h1 = h2 = 0 can result in a final output of anything between [0, 1].

The second difference is that the hidden nodes and output nodes now have two halves to them. This is quite an uncommon way to draw out nodes but it'll help to explain the concept of *activation functions*. HAi represents the *activation function* of hidden node i.

After multiplying the inputs by their respective weights and summing them into the hidden nodes, each hidden node will run an *activation function* f() to provide an output to the nodes in the next layer. So the output of H1 will be

> HA1 = f(I1 x W1 + I2 x W3)

and the output of H2 will be

> HA2 = f(I1 x w2 + I2 x w4)

The output of the output nodes also undergo an activation function.

The final difference is that now there are two output nodes. One output node suffices for *binary classification* (think hotdog not hotdog) but can get clunky if you try to fit in more than two possible outcomes. For example: having an output of 
>[0 .33] = hotdog, (.33, .67) = burger, [.67 1] = sandwich

is worse for the people and machine involved as opposed to having 1 output node for each outcome. There are also mathematical reasons behind this: what if an item was half sandwich half hotdog? These two items are at the ends of the distribution and as such the machine might output a value that is in the middle, completely misclassifying the item as a burger. The reason why this happens is because there is no linear relationship between hotdogs, burgers, and sandwiches. Burgers aren't necessarily half hotdog and half sandwich so representing all three on a single scale of 0 to 1 is incorrect. This idea is important for structured data and categorical variables (though we won't be using these for our project).

Having multiple output nodes increases the flexibility in what our machine can tell us about the data. For image recognition, networks have an output node per category and this allows the network to express how likely an image is multiple things. In this example, OA1 would tell you how much the machine believes its a hotdog, and OA2 would tell you much the machine believes its a burger. If, continuing the example, both OA1 and OA2 return 1, we can conclude that the network believes the image is a hotdog and a burger i.e. the classifier cannot tell if it is one or the other.

## Activation Functions
Some useful links if you don't want to read this section:

* [Understanding Activation Functions](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0)
* [A Practical Guide to Relu](https://medium.com/tinymind/a-practical-guide-to-relu-b83ca804f1f7)
* [Activation Functions Cheatsheet](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html)

Some activation functions:

* Sigmoid
* Tanh
* ReLu (and its variations)
* Softmax 

Neural networks are inspired by biological neurons. These neurons take inputs and "fire" a signal to its output neurons. Whether it a signal is fired or not depends on how strong the inputs are. Likewise, activation functions determine whether or not a node in a neural network fires a signal and how strong that signal is to its downstream nodes.

A traditional activation function is the sigmoid:

![sigmoid](https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Sigmoid-function-2.svg/2000px-Sigmoid-function-2.svg.png)

It's smooth and limits the output to be within (0, 1). So if a node had an input of 10, its output would be very close to 1 using the sigmoid activation.

Here is the sigmoid (labeled as logistic) plotted against the tanh and linear activation functions:

![3actfuncts](https://theclevermachine.files.wordpress.com/2014/09/nnet-error-functions2.png?w=700&h=352)

Note that linear activation isn't a great function to use in most cases. Also note that tanh is a transformed sigmoid like so:

> tanh(x) = 2 sigmoid(2x) - 1

Why is linear activation not so great? Let's pay attention to the derivatives in the above image. The derivatives of activation functions are used in an optimization technique called *gradient descent*. Basically, we're searching for the minima of a function and we're using the slope of the function as a way to figure out which direction to move in. In a 2d graph we'd either be moving left or right (increase or decrease x). 

Linear functions have the same derivative regardless of its input, thus the algorithm cannot tell which direction to modify the inputs to improve. (Don't think too hard on this as we'll be going into more detail later)

However this isn't the biggest issue: the biggest issue is of the mathematical sort. In a deep neural network, we can think of each hidden layer as a function that takes input from the previous layer and outputs to the next layer. We can think of this as "stacking" layers to solve a problem, which be though of as compositing functions to approximate some solution. When we composite linear functions, what we get at the end is another linear function. It doesn't matter how many linear functions you "stack" you will get a linear function at the end of it all. This severely limits what sort of problems we can solve. For more info read the article on the [Universal Approximation Theorem](https://towardsdatascience.com/can-neural-networks-really-learn-any-function-65e106617fc6).

So what's the deal with ReLU then if linear activations aren't so great? Lets take a look at it's graph:

![ReLU](https://www.researchgate.net/publication/323956667/figure/fig1/AS:607180243861515@1521774459242/The-Rectified-Linear-Unit-ReLU-activation-function-produces-0-as-an-output-when-x-0.png)

While the positive side is linear, the negative side is not. This introduces non-linearity into an otherwise straight line. So ReLU doesn't deal with the same problem as a naive linear activation. However there are still issues with ReLU.

Let's pay attention to the negative side of ReLU's graph. The slope there is 0 regardless of close the input is to 0 when the input is negative. This can result in something called "dead" neurons. When a neuron begins to receive negative input, it'll be unlikely that the neuron will ever recover since gradient descent can only detect that the slope is 0 (it can't tell which direction to adjust the input weights). For more info read [A Practical Guide to ReLU](https://medium.com/tinymind/a-practical-guide-to-relu-b83ca804f1f7).

However the non-saturating nature of ReLU does provide a substantial boost to training speed. See this [paper](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf) page 3. Non-saturating means the activation function does not "squeeze" the input like a sigmoid or tanh does. Non-saturating functions are also resilient to the vanishing gradient problem. Read more [here](https://adventuresinmachinelearning.com/vanishing-gradient-problem-tensorflow/).

This image graphs the variants of ReLU created to address the dead neuron issue. Notice there are now non-zero slopes associated with the negative half of the space. From a probability perspective we're bringing the average output of the activation function closer to 0.

![ReLU Variants](https://i0.wp.com/laid.delanover.com/wp-content/uploads/2017/08/elu.png)

Finally, an important activation function for multi-class classifiers is [softmax](https://en.wikipedia.org/wiki/Softmax_function), which outputs a categorical distribution - a probability distribution over *K* different possible outcomes. So given multiple outputs, softmax takes these and turns them into values that are within [0, 1] and sum up to 1. We basically get the percentage of how confident the network is for each possible output. Example: [0.87 hotdog, 0.06 burger, 0.07 sandwich]


## Layers

Lets briefly talk about layer types. We'll be going over fully connected, convolution, and subsampling. For more information on other layers check out [A mostly complete chart of neural networks explained](https://towardsdatascience.com/the-mostly-complete-chart-of-neural-networks-explained-3fb6f2367464)

![cnn_layers](https://www.researchgate.net/profile/Yann_Lecun/publication/220320307/figure/download/fig1/AS:305682390765571@1449891769454/Architecture-of-convolutional-network-used-for-training-This-represents-one-slice-of-the.png)

The image above is a basic *Convolutional Neural Network*. These are typically used for image recognition. Lets just focus on the layers for now: fully connected, convolution, and subsampling.

#### Fully Connected
![Fully connected](https://i.stack.imgur.com/BVZro.png)

#### Convolution

Convolution layers are great for extracting information from image data. This is done by running a matrix multiply across the input data with a filter. For example if we had a 28x28 image we could run a 3x3 filter left to right and top to bottom. These filters are also learnable by the machine. See below for an example:

![Convolution Layer](https://i.gyazo.com/c6efc0e9b577d5b47acecadbc134a7d8.png)

You can see this animated [here](http://cs231n.github.io/convolutional-networks/#conv) (scroll down a bit).
Here we have 3 layers of input. This is because pixels are stored as 3 values for RGB. There may be multiple filters per convolution layer. There are also parameters such as *stride*, which defines how many cells to move the filter by on each calculation, and *padding*, which sets a border of 0's around the input to allow the filters to run across the input's edges as well.

Check out this [great video of a basic convolutional network](https://www.youtube.com/watch?v=f0t-OCG79-U) or CNN for a visual of a basic CNN in action. Note that the video also includes subsampling layers after every convolution layer and a fully connected layer at the end.

* [0:00 - 0:34] Convolution Layer 1 with filter 1 (padding and stride of 1)
* [0:34 - 0:57] Convolution Layer 1 with filter 2
* [0:57 - 1:01] ReLU
* [1:01 - 1:09] Subsampling
* [1:09 - 1:17] Convolution Layer 2 with filter 3
* [1:17 - 1:26] Convolution Layer 2 with filter 4
* [1:26 - 1:28] ReLU
* [1:28 - 1:32] Subsampling (couldn't tell if this was avgpool or maxpool. It seems more like avgpool)
* [1:32 - 1:43] Fully-connected (I assume they ran a softmax as well)

#### Subsampling (or pooling layers)

The video in the previous section also included subsampling layers. Subsampling is used to reduce the dimensions of the input. In other words, we're reducing the "resolution" of the features we obtained from the previous layers. In the case of a CNN we subsample the features learned through the convolution layers.

Two popular pooling functions are maxpool and averagepool. Maxpool simply returns the max value of a submatrix, while average pool returns the average of a submatrix. See the image below for an example of maxpool.

![maxpool](http://cs231n.github.io/assets/cnn/maxpool.jpeg)

Subsampling helps with generalization and in image recognition they help to not only increase training speed (by reducing the dimensionality of the feature space) but also make the model more resilient against shifts and distortions.

#### Linear activation layers

This is a sidenote. PyTorch will refer to activations (sigmoid and ReLU) as layers.

#### Other layers

There are many other types of layers. Read this article [Residual Learning](https://cdn-images-1.medium.com/max/1600/1*pUyst_ciesOz_LUg0HocYg.png) if you want to learn more on residual learning. A network we'll be looking at called resnet34 will utilize residual layers (no residual layer knowledge required). Another interesting layer type are [1x1 convolutions](https://iamaaditya.github.io/2016/03/one-by-one-convolution/). [A mostly complete chart of neural networks explained](https://towardsdatascience.com/the-mostly-complete-chart-of-neural-networks-explained-3fb6f2367464) also goes into a variety of different network types that may interest you.


## Optimization
Some useful links if you don't want to read this section:

* [Gradient Descent Cheatsheet](https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html)
* [Loss Functions Cheatsheet](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html)
* [Common Loss Functions in Machine Learning](https://towardsdatascience.com/common-loss-functions-in-machine-learning-46af0ffc4d23) 

So we now know that neural networks are a series of node layers. Each node takes in inputs and computes the activation function which is then sent as input to the next layer. Each connection from a node to the next is defined with a weight value that determines how much impact a node has on a node in the next layer. Now let's talk about how the algorithm learns what weights solve the problem.

Before the algorithm can optimize it needs to know what to optimize. Let us first define a *loss function*. A loss function tells us (and the algorithm) how good the model is. For more info read [Common Loss Functions in Machine Learning](https://towardsdatascience.com/common-loss-functions-in-machine-learning-46af0ffc4d23)

Something to take note of is that we do not optimize the loss function *directly*. i.e. we do not change the loss function itself. We change the *parameters* i.e. the *weights* of the model in the hopes that we improve the loss function's result. This is a difference between learning and pure optimization.

![gradient_descent_2d](https://cdn-images-1.medium.com/max/1600/0*rBQI7uBhBKE8KT-X.png)

So we have a loss function, which can be visualized as a curve. We need to know in which direction to adjust the weights such that the loss function is minimized. This is where gradient descent comes in. If you recall from calculus, a gradient is the derivative of a multi-variable function. Each weight here is a variable so when we have large networks our gradients are also large in terms of complexity.

>Given f(x, y) the gradient is: (df/dx, df/dy)

Remember that each layer depends on the input from another layer so our gradients can become difficult to compute as we increase the number of layers. You may have heard the term *backpropagation*. In machine learning, backpropagation is a clever implementation of the chain rule and is used to quickly calculate the gradient of neural networks. You can blackbox backpropagation. If you are curious, however, read [Michael Nielson's post on backpropagation](http://neuralnetworksanddeeplearning.com/chap2.html)

Now that we have calculated the gradient, we can "see" in what direction adjust our parameters to reduce our cost function. Watch [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) for a comprehensive explanation. See the below image for an example of gradient descent in 3d:

![gradient_descent_3d](https://www.sciencemag.org/sites/default/files/styles/inline__450w__no_aspect/public/ma_0504_NID_alchemy_WEB.jpg?itok=OzWhcdrK)

Now how does gradient descent actually run? We know that we have a loss function and we compute the gradient to understand how to adjust our parameters. When do we adjust the parameters? How much do we adjust them by?

Let us assume our network is tasked with learning to distinguish between hotdog not hotdog. Typically gradient descent begins by initializing our parameters randomly and then running each data sample through the model. It then computes the loss function - basically outputs some number that indicates how accurate the model is by comparing the prediction vs. the actual label. It then utilizes backpropagation to calculate the gradient - how does each weight affect the accuracy of our model - and adjusts the weights to make the model more accurate. Rinse and repeat.

So in our example we adjust the parameters after running through the entire dataset. We make the network compute the prediction of every datapoint before "learning" from our results. This can get extremely expensive when datasets are large. Introducing *stochastic* gradient descent [here](https://developers.google.com/machine-learning/crash-course/reducing-loss/stochastic-gradient-descent) and [here](http://deeplearning.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/). 
Stochastic Gradient Descent optimizes by updating the weights or "learning" after a subset of data - called a *batch* - as opposed to the entire dataset. SGD also has the advantage of being more stable during training. Here's an analogy as to why: imagine a professor who gives feedback to a student on every mistake they make as opposed to every 1000 mistakes. It'll be much easier for the student to learn from 1 mistake as opposed to 1000 at a time.

So now we know when we update the weights. What about how much do we change them? See the below image for a visualization:

![Learning_Rate_Image](https://cdn-images-1.medium.com/max/1600/0*QwE8M4MupSdqA3M4.png)

The *learning rate* is exactly that: how much our algorithm adjusts its weights or learn with each step. With a large rate we risk divergent behavior, an example of which is where we bounce back and forth and never reach a minimum. With a small rate we waste computational hours by taking longer to train our models. Introducing learning rate annealment.

![Learning_Rate_Annealment](https://cdn-images-1.medium.com/max/1600/1*iSZv0xuVCsCCK7Z4UiXf2g.jpeg)

An *epoch* is when the entire dataset has been learned from once. So epoch 20 means we cycling through the data for the 20th time. Note that this is different from batches, which signify the amount of data we cycle through before we update the model.

Learning rate annealment is when we adjust our rate as time goes on. The rationale being that when we first begin the algorithm we do not have a good sense of where the optimum is so we search far and wide. As time goes on and the model starts to converge, we reduce our learning rate as a way to finesse the model into an optimum. To read more in depth visit [here](http://cs231n.github.io/neural-networks-3/#anneal).

What if there are multiple minimums and we get stuck in one of them? Like in this picture:

![stuck](https://cdn-images-1.medium.com/max/800/1*0k0dUpICE5BJe6VQFmBLJA.png)

Sometimes we like to increase the learning rate again in the hopes that we find a better minimum like so:

![sgdr](https://cdn-images-1.medium.com/max/800/1*5T8mc-cCBabeGYVSG3VPBw.png)

We can do this with [Stochastic Gradient Descent with Restarts](https://towardsdatascience.com/https-medium-com-reina-wang-tw-stochastic-gradient-descent-with-restarts-5f511975163). Basically with this version of SGD the learning rate now looks like this:

![sgdr_lr](https://cdn-images-1.medium.com/max/1200/1*nBTMGa3WqhS2Iq4gCeCZww.png)

In this example each subsequent learning decay period lasts longer. The rationale is similar to what motivated learning rate annealing in the first place: get a quick overview of the terrain and then slowly converge into a minima.

There are other optimization methods such as [Adam](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/) (and the fixed version [AdamW](https://www.fast.ai/2018/07/02/adam-weight-decay/)) and [Stochastic Gradient Descent with Momentum](https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d). Feel free to read up on these on your own time.

## Generalizing Models

So far we've learned what neural nets are, what math functions they consist of, and how they learn. Now let's talk about how sometimes our models can *overfit* the problem. Overfitting is when a model is so well tuned to the training data that it fails to generalize to real or test data. We want out models to do well with data it hasn't seen before and this is called the model's ability to *generalize*.

An example of a neural network overfitting is if it can distinguish dogs but only if its long-haired, in a sitting position, and is light colored. It fails to generalize and would not recognize a running chihuahua as a dog. There can be many reasons as to why this particular model has overfit.

![overfit](https://cdn-images-1.medium.com/max/1125/1*_7OPgojau8hkiPUiHoGK_w.png)

Overfitting can occur when we:

1. Over-train or train too long
2. Utilize a network that's too large for the dataset
3. Use a non-comprehensive data set

When we train for too long, the network starts to pay attention to progressively smaller and more minute details of the data that may not have much to do at all with the output. It starts to learn all the specifics of the given dataset as opposed to learning the intuition behind it. 

Second, utilizing larger and larger networks also motivate overfitting. Since the network is more flexible it can bend it's solution into more esoteric details of the data as opposed to the big ideas or again, the intuition behind the data. 

Finally, using a bad data set (for example using a set of only golden retrievers to teach a network to recognize dogs in general) for a given problem will handicap the network from generalizing. This list is not comprehensive and there are other reasons behind overfitting.

So what are some techniques we can use to prevent overfitting?

1. smaller networks
2. data augmentation
3. dropout
4. regularization

Smaller networks force the system to learn just the big details as opposed to all the smaller nuances of the data set. This should be last resort since smaller networks also decrease a model's potential to learn and solve complex problems.

Data augmentation is the creation of new data to supplement existing data. An example of this is flipping, rotating, and skewing images in a dataset to provide variation. This works particularly well with image classification. However, we probably wouldn't want to vertically flip images when training a model to recognize dogs vs cats as images of them are almost always upright. We would want to vertically flip images for tasks such as recognizing cancer cells or classifying satellite images.

Dropout is literally the dropping out of node activations when running a model. The program will ignore neurons with some probability, forcing the network to learn robust features of the data. Thus, given noise the model will still be able to perform well.

Regularization is a general term for "any modiﬁcation we make to a learning algorithm that is intended to reduce its generalization error but not its training error" (deep learning book). Training error here indicates how well the model performs on the given data set to learn from. Regularization techniques range from penalizing large weights to other constraints in the model. Read in depth [here](https://www.deeplearningbook.org/contents/regularization.html).

Okay great. We understand what overfitting is and how to combat it. But how do we measure overfit?

Introducing hold-out sets AKA the training set, validation set, and test set. In deep learning (and machine learning in general) we split our data into different sets for different tasks. There are a variety of data techniques in machine learning such as bootstrapping and k-fold cross validation but for the purpose of this project we'll only be concerned with training, validation, and test sets.

Lets define what they're used for. Training sets are the data that our models will learn from. It's the data that is used by the optimizer algorithm for every epoch. Validation sets are a way to measure overfit (we'll see how in a moment). They also assist in selecting hyperparameters for our model. Finally, test sets are set aside to judge our final solution. We never use the test set until the very end. This is useful for when we want to compare different models against each other.

So how do use validation sets to measure overfit? First we train our models on the training data. We receive some value for our loss (the lower the loss the more accurate our model is). We then run the validation data through our model but the model *does not update* its weights from this data, which means the model isn't learning from the validation data. We are only interested in the loss value of the model's results on the validation set.

![loss_table](https://i.gyazo.com/c8c145367ba7ddaade836c5c17abc0bc.png)

In the loss table above we can see that by epoch 20 we've overfit. We can tell because while our loss on the training set is low, our validation loss is higher. We can conclude that while the model is performing well on the training data, it is failing to perform as well on the validation data, thus we are not generalizing well. Note that in some cases we can overfit to the validation set as well though it is not as likely as training overfit.

An example of this occurs on Kaggle, the data science competition website. Here, users get graded on a public and private test set. The public test set serves as a heuristic for users on how their model is performing. The private test set is for the final judgement (1st place, 2nd place, etc...). What ends up happening is people will make many submissions and over time overfit their models to the public leaderboard. When the private leaderboard is released they see that they have dropped many places in the ranking because their model has overfit.

## Dealing with Data
This (optional) section will cover how to split the data into training, validation, and test sets, normalizing data, how to deal with structured data, and how to deal with sequential data.

Disclaimer: Theres a great deal more to know when it comes to working with data. This only covers the basics for deep learning. See [here](https://data.berkeley.edu/degrees/data-science-ba) for more.

#### Splitting Data

Splitting data is relatively straight-forward. We generally want something about a 70:20:10 split or a 60:20:20 split. When dealing with a multi-class classificiation problem, we want to make sure that within each set, there are the same amount of instances of each class. So for example we would have 70 cats, 70 dogs, and 70 elephants in the training set, 20 of each in the validation set, and 10 of each in the test set.

What happens if you don't have enough instances of one class to go around? You could oversample - duplicate instances of the smaller class so that the number of instances per class are more even. The downside is that this reduces the variance of this class which increases the risk of asymmetric overfitting.

What happens if the data is in the form of a time series? We would want the training set to the be the earliest set of data points, the validation in the middle, and the test set to be the most recent set of data points. Why? This is so that our validation and test set show us how well our models are predicting the future datapoints.

#### Normalizing Data

Let's say we have two columns of data that we want to feed into a neural net: price and age. Let us assume that we're trying to predict the class of big-ticket musical instruments. An example is a Stradivarius violin. These are approximately 300 years old and are worth millions. When we pass these values into the model, it has no concept of age and money. All it knows is that there is a large 7-figure value and a smaller 3-figure value. Mathematically, which scalar will have a larger affect on the output? Let's assume as an example that the age is much more significant than price in determining the label of our data sample. This means that the machine must learn that the smaller 3 figure age values are of greater significance than the 7 figure price value. In this example it shouldn't be too hard: there are only 2 columns. But imagine if the machine had to do this with hundreds of columns.

In order to combat this we preprocess the data by *normalizing* it. *Normalizing* means we transform the data such that the mean is 0 and the standard deviation is 1. This is done by subtracting the mean from every datapoint in a column and then dividing every datapoint by the standard deviation of said column. This process preserves the information relative to itself while making it easier for machine learning algorithms to learn from. Now on the initialization of a machine learning model every data column will have approximately the same significance to the output relative to each other.

There are some caveats when performing this operation however. First of all is ensuring that the normalization is done uniformly across data sets. If you normalize the training set using the mean and standard deviation derived from said set, you must ensure that you normalize your validation and test set with the same values! Do not normalize the training set based on the training mean and std and then normalize the validation set based on validation mean and std. This would be akin to translating the data two different ways and expecting our model to know the difference without telling it.

What we've talked about thus far takes care of the data distribution going into our input layers. What about hidden layers? The weights and activation functions of each layer change the distribution of the data as we feed it forward through the network. This can slow down the training process since every layer now has to learn from a different distribution at every step as opposed to a normalized distribution. We call this problem *internal covariate shift*. To solve this problem we implement something called *batch-normalization*. We won't be going into details on it within this document. Read [here](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/batch_norm_layer.html), [here](https://towardsdatascience.com/batch-normalization-theory-and-how-to-use-it-with-tensorflow-1892ca0173ad) and [here](https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c) for more info.

#### Structured Data

Up to this point we've been talking about *unstructured* data - data that is not organized in a predefined manner. For example the RGB values in an image are unstructured data. So are the speech waveforms of the human voice. *Structured* data is the opposite. Things like the day of the week, merchandise category, movie genre, etc.... Another way to think about data is between categorical and numerical variables.

Handling unstructured data seems straightforward. Everything is the same datatype so just throw it all into a neural network and let the neural network do its work. Structured data is a bit more complex to handle. We can't just have Sunday be 0 and Saturday be 6. There's no mathematical relationship that defines Saturday as 6 times Monday.

An initial solution to handling this is something called *1-hot encoding*. This means that for every categorical variable we have an input node for every possible value of that category. For example, for days of the week we'll have 1 node for is-Sunday, 1 node for is-Monday, etc.... While this does solve the problem of encoding structured data there are two issues. 

First of all, what happens when we have categorical data with a large cardinality? I.e. a large amount of possible values. For example, the English language has over 170,000 words in use. Do we create an input node for each word? This is inefficient. Second of all, we lose the underlying relationships between the values of the category. For example, Saturday and Friday would be very similar since they're both days people can sleep late and Saturday and Sunday would be very similar since they're both weekend days. Using 1-hot encoding doesn't allow the network to learn relationships between the category's values.

This is where *Embeddings* come in. *Embeddings* are simply mathematical representations of a concept. For example, we can use a 4 dimensional vector to represent the day of the week, which means that to the machine, the day of the week will be represented as 4 numbers and it is up to the machine to learn how to encode these numbers for each day. For example, Sunday might be [0, 1, 1, 1] and Saturday might be [0, 1, 1, 0]. These two are close in 4 dimensional space and this represents their similarity as weekend days. Instead of 7 nodes, we now use 4.

Embeddings are also used in NLP (Natural Language Processing) as a way to mathematically represent words. You can download pretrained embeddings to utilize in your own models; their dimensionality tends to range up to 500 (quite impressive for representing over 100,000 words).

#### Sequential Data

Everything we've talked about so far can be applied to networks that only need to worry about one data point at a time. One image of a cat does not affect the meaning behind another image of a cat. However, this is quite different from language where the order of words matter. For example: 

>I love eating green eggs and ham.

vs. 

>ham I eating green eggs love and.

Clearly proper order conveys proper meaning.

There's a machine learning method for NLP called bag of words. Basically, we count the number each word appears and try to extrapolate meaning from that. The model performs satisfactorily for simple tasks. However, when we decide to count bigrams - sequence of two adjacent elements - instead of single words, the performance of bag of words increases. This is useful in character based languages as well such as mandarin and japanese. From this we can surmise that empirically, preserving sequential information increases AI performance.

But now we want to go bigger than bag of words. We're working with *deep* neural networks. Introducing recurrent networks:

![RNN](https://www.researchgate.net/profile/Ramon_Quiza/publication/234055140/figure/fig3/AS:299964002521099@1448528399394/Graph-of-a-recurrent-neural-network.png)

Notice the connections from the output layer *back* to the hidden layer. Unlike *feed-forward networks* (networks that only move in one direction), this network sends data back into the network. This is because the word that came before the current word can vastly alter the overall phrase's meaning. For example: not good vs. quite good.

For more checkout [An Intro to RNNs](https://medium.com/explore-artificial-intelligence/an-introduction-to-recurrent-neural-networks-72c97bf0912).

## Resources

TODO: Copy/Paste all links here.

I used a lot of links throughout this doc and only some are listed here. Some links here weren't used in the doc.

### Comprehensive Resources
* [Beginner's Guide to Neural Networks and Deep Learning - skymind](https://skymind.ai/wiki/neural-network)
* [Neural Networks and Deep Learning - Michael Nielson](http://neuralnetworksanddeeplearning.com) - a good general overview
* [Deep Learning - Ian Goodfellow, Yoshua Bengio, Aaron C. Courville ](https://github.com/janishar/mit-deep-learning-book-pdf) - a one-stop comprehensive resource

### Videos
* [What is a neural network](https://www.youtube.com/watch?v=aircAruvnKk)
* [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w)

### Topical Resources
* [Quick intro to neural networks](https://ujjwalkarn.me/2016/08/09/quick-intro-neural-networks)
* [Backpropagation Algorithm](http://colah.github.io/posts/2015-08-Backprop/)
* [Batch-norm](https://www.quora.com/What-is-a-batch-norm-in-machine-learning)
* [Structured Data in Deep Learning](https://towardsdatascience.com/structured-deep-learning-b8ca4138b848)

### Optional Math Resources
* [Review of Probability Theory - CS229 ML @ Stanford](http://cs229.stanford.edu/section/cs229-prob.pdf)
* [Review of Linear Algebra - CS229 ML @ Stanford](http://cs229.stanford.edu/section/cs229-linalg.pdf)
* [Summary of Maths for CS189/289 ML @ Berkeley](http://gwthomas.github.io/docs/math4ml.pdf)

# Introduction to Deep Learning
Deep learning is a deep topic. Let's Dive in.

## Table of Contents
* Overview
* A bit deeper
* Resources

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

## A Bit Deeper
Some useful links if you don't want to read this section:

* [What is a neural network](https://www.youtube.com/watch?v=aircAruvnKk)
* [Quick intro to neural networks](https://ujjwalkarn.me/2016/08/09/quick-intro-neural-networks)

![simplerNN](https://i.gyazo.com/511d2bbf4aee74c1be4afcd4ed847ea7.png)
Above is a simple Neural Network. There are two input, two hidden, and one output node. Information flows from the input nodes to the hidden nodes and finally to the output node. w stands for weight: they represent how much of an effect an input has on the next node. 

The input to node h1 is the sum of x1 \times w11 and x2 \times w21. The input to node h2 is the sum of x1 \times w12 and x2 \times w22.

This allows the network to learn *representations* or *features* inherent in the data. Note that this means that biases can also be learned in the data, which is why good and well-formed data is integral to a good model. The learning algorithms are only as good as the data we provide it.

The input to the output node is the sum of h1 \times w3 and h2 \times w4.

This allows the network to combine its knowledge of features to decide on an output. In deeper networks, or networks with more layers, the algorithm can learn simple representations and along each layer combine them into more complex representations. An example in image recognition is the network can recognize lines, which combine into shapes, which combine into small objects, which finally combine into the main subject of the image.

![simpleNN](https://dzone.com/storage/temp/7913025-neural-network.png)

## Resources
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
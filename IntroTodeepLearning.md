
# Introduction to Deep Learning
Deep learning is a deep topic. It also happens to have a large vocabulary so don't fret if you feel overwhelmed: keep calm and ask google. Let's dive in.

## Table of Contents
* Overview
* Basic Neural Network
* Activation Functions
* Optimization
* Generalizing Models
* Dealing with data
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

![simpleNN](https://dzone.com/storage/temp/7913025-neural-network.png)

This above image introduces some new things from the previous image. 

There is now an extra node in each layer called a *bias node* that will always output 1. The weights for the bias nodes determine how strong the bias is. This provides flexibility to the network by providing a value that is independent of the input. For example, let's assume all nodes can have some value between 0 and 1. If the output of h1 and h2 were 0, the network's output would always be 0 in this case since any weight multiplied by 0 is 0. By introducing the bias node, h1 = h2 = 0 can result in a final output of anything between [0, 1].

The second difference is that the hidden nodes and output nodes now have two halves to them. This is quite an uncommon way to draw out nodes but it'll help to explain the concept of *activation functions*. HAi represents the *activation function* of hidden node i.

After multiplying the inputs by their respective weights and summing them into the hidden nodes, each hidden node will run an *activation function* f() to provide an output to the nodes in the next layer. So the output of H1 will be

> HA1 = f(I1 x W1 + I2 x W3)

and the output of H2 will be

> HA2 = f(I1 x w2 + I2 x w4)

The output of the output nodes also undergo an activation function.

The final difference is that now there are two output nodes. One output node suffices for *binary classification* (think hotdog not hotdog) but can get clunky if you try to fit in more than two possible outcomes. For example: having an output of 
>[0 .33] = hotdog, (.33, .67) = burger, [.67 1] = sandwich

is worse for the people and machine involved as opposed to having 1 output node for each outcome. There are also mathematical reasons behind this: what if an item was half sandwich half hotdog? These two items are at the ends of the distribution and as such the machine might output a value that is in the middle, completely misclassifying the item as a burger. The reason why this happens is because there is no linear relationship between hotdogs, burgers, and sandwiches. Burgers aren't necessarily half hotdog and half sandwich so representing all three on a single scale of 0 to 1 is incorrect. This idea will be important for structured data and categorical variables (though we won't be using these for our project).

Having multiple output nodes increases the flexibility in what our machine can tell us about the data. For image recognition, networks have an output node per category and this allows the network to express how likely an image is multiple things. In this example, OA1 would tell you how much the machine believes its a hotdog, and OA2 would tell you much the machine believes its a burger. If, continuing the example, both OA1 and OA2 return 1, we can conclude that the network believes the image is a hotdog and a burger. I.e. the classifier cannot tell if it is one or the other.

## Activation Functions
Some useful links if you don't want to read this section:

* [Understanding Activation Functions](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0)
* [A Practical Guide to Relu](https://medium.com/tinymind/a-practical-guide-to-relu-b83ca804f1f7)
* [Activation Functions Cheatsheet](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html)

Pay attention because these are important.

* Sigmoid
* Tanh
* ReLu (and its variations)
* Softmax 

Neural networks are inspired by biological neurons. These neurons take inputs and "fire" a signal to its output neurons. Whether it a signal is fired or not depends on how strong the inputs are. Likewise, activation functions determine whether or not a node in a neural network fires a signal and how strong that signal is to its downstream nodes.

A common activation function is the sigmoid:

![sigmoid](https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Sigmoid-function-2.svg/2000px-Sigmoid-function-2.svg.png)

It's smooth and limits the output to be within (0, 1). So if a node had an input of 10, its output would be very close to 1 using the sigmoid activation.

Here is the sigmoid (labeled as logistic) plotted against the tanh and linear activation functions:

![3actfuncts](https://theclevermachine.files.wordpress.com/2014/09/nnet-error-functions2.png?w=700&h=352)

Note that linear activation isn't a great function to use in most cases. Also note that tanh is a transformed sigmoid like so:

> tanh(x) = 2 sigmoid(2x) - 1

Why is linear activation not so great? Let's pay attention to the derivatives in the above image. The derivatives of activation functions are used in an optimization technique called *gradient descent*. Basically, we're searching for the minima of a function and we're using the slope of the function as a way to figure out which direction to move in. In a 2d graph we'd either be moving left or right (increase or decrease x). 

Linear functions have the same derivative regardless of its input, thus the algorithm cannot tell which direction to modify the inputs to improve. (Don't think too hard on this as we'll be going into more detail later)

However this isn't the biggest issue. The biggest issue is of the mathematical sort. In a deep neural network, we can think of each hidden layer as a function that takes input from the previous layer and outputs to the next layer. So, we "stack" layers to solve a problem, which can be rephrased as we composite functions to approximate some solution function. When we composite linear functions, what we get at the end is another linear function. It doesn't matter how many linear functions you "stack" you will get a linear function at the end of it all. This severely limits what sort of problems we can solve. For more info read the article on the [Universal Approximation Theorem](https://towardsdatascience.com/can-neural-networks-really-learn-any-function-65e106617fc6).

So what's the deal with ReLU then if linear activations aren't so great? Lets take a look at it's graph:

![ReLU](https://www.researchgate.net/publication/323956667/figure/fig1/AS:607180243861515@1521774459242/The-Rectified-Linear-Unit-ReLU-activation-function-produces-0-as-an-output-when-x-0.png)

While the positive side is linear, the negative side is not. This introduces non-linearity into an otherwise straight line. So ReLU doesn't deal with the same problem as a naive linear activation. However there are still issues with ReLU.

Let's pay attention to the negative side of ReLU. The slope there is 0 regardless of close the input is to 0 when the input is negative. This can result in something called "dead" neurons. When a neuron begins to get negative input, it'll be unlikely that the neuron will ever recover out of that since gradient descent can only see that the slope is 0 (it can't tell which direction to adjust the input weights). For more info read [A Practical Guide to Relu](https://medium.com/tinymind/a-practical-guide-to-relu-b83ca804f1f7).

This image graphs the variants of ReLU created to address this issue with ReLU. Notice there are now non-zero slopes associated with the negative half of the space.

![ReLU Variants](https://i0.wp.com/laid.delanover.com/wp-content/uploads/2017/08/elu.png)

Finally, another activation function for multi-class classifiers is [softmax](https://en.wikipedia.org/wiki/Softmax_function), which outputs a categorical distribution - a probability distribution over *K* different possible outcomes. So given multiple outputs, softmax takes these and turns them into values that are within [0, 1] and sum up to 1. We basically get the percentage of how confident the network is for each possible output. Example: [0.87 hotdog, 0.06 burger, 0.07 sandwich]

## Gradient Descent



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
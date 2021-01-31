# DEEP LEARNING 101

![logo](https://github.com/abhishek-pes/DL/blob/main/Assets/Machine-Learning.gif)

## WHAT IS DEEP LEARNING

In simple terms, Deep learning is a branch of machine learning which is characterised by deep stacks of machine learning.
But what makes it different from traditional ML models is that **in deep learning, the algorithm is given raw data and decides for itself what features are relevant.**

Through their power and scalability neural networks have become the defining model of deep learning. Neural networks are composed of neurons, where each neuron individually performs only a simple computation. The power of a neural network comes instead from the complexity of the connections these neurons can form.

## A SINGLE NEURON

The most fundamental component of a neural networs is a **Single Neuron**, also called as a **Unit**.

> The below diagram is a representation of a single neuron.

![single neuron](https://github.com/abhishek-pes/DL/blob/main/Assets/image.png)

Here the input is `x`. Its connection to the neuron has a weight which is `w`. Whenever a value flows through a connection, you multiply the value by the connection's weight. **For the input x, what reaches the neuron is w \* x**.

Now how does the neural network actually learn?
the answer is quite simple, **a neural networks by modifying its weights**

## NEURAL NETWORK

when many neurons are connected together it forms a neural network. In a neural network the neurons are organised into layers and each **layer** performs a relatively simple calculation.

![neural network](https://github.com/abhishek-pes/DL/blob/main/Assets/nerual%20network.JPG)

When we collect together linear units having a common set of inputs we get a **dense layer**.

There are many kind of layers. A layer can be, essentially, any kind of data transformation. Many layers, like the convolutional and recurrent layers, transform data through use of neurons and differ primarily in the pattern of connections they form. Others though are used for feature engineering or just simple arithmetic.

https://www.tensorflow.org/api_docs/python/tf/keras/layers

## HOW DEEP LEARNING WORKS

Deep Learning Algorithms use something called a neural network to find associations between a set of inputs and outputs. The basic structure is seen below:

![dl working](https://github.com/abhishek-pes/DL/blob/main/Assets/how%20dl%20works.jpg)

A neural network is composed of input, hidden, and output layers — all of which are composed of “nodes”. Input layers take in a numerical representation of data (e.g. images with pixel specs), output layers output predictions, while hidden layers are correlated with most of the computation.

Information is passed between network layers through the function shown above. The major points to keep note of here are the tunable weight and bias parameters — represented by w and b respectively in the function above. These are essential to the actual “learning” process of a deep learning algorithm.

![activation functoin](https://github.com/abhishek-pes/DL/blob/main/Assets/activation%20function.png)

After the neural network passes its inputs all the way to its outputs, the network evaluates how good its prediction was (relative to the expected output) through something called a loss function.
**The goal of the neural network is to ultimately reduce this loss function by changing the parameters of w and b**

## CONCLUSION

Deep learning is ultimately an expansive and complex field. Various types of neural networks exist for different tasks (e.g. Convolutional NN for computer vision, Recurrent NN for NLP).
The implications of deep learning are insane. Video synthesis, self driving cars, human level game AI, and more — all of these came from deep learning.

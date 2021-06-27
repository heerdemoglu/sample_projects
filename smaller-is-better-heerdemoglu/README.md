# Smaller is better!

As you probably already know, Plexagon is a mobile app publisher 📱 that focuses on creating the best tools for content creators 🌅. Our loyal users recently suggested that we create a fascinating app for recognizing what their friends are wearing 👘, and we have decided to make them happy! 👾


## Dataset & model

While browsing on the internet, we stumbled upon the [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist), which looks like an excellent fit for what we are trying to achieve. We were also lucky enough to find a [model](fashion_mnist.py) ready to be used for training our dataset that works very well (>90% accuracy).


## Our problem

Unfortunately, after giving the output model to our engineering team, we figured out that there were a few issues with it:
1) The model has too many parameters (roughly 1.6M) and it's too slow to use for production purposes in real-time.
2) The model is too big in terms of disk size, and we are not able to embed it on an iPhone.
3) The model is not in a supported format, as our iOS app only supports models in Tensorflow Lite/CoreML/PyTorch mobile models.


## Your challenge

We would like to understand if there is an intelligent way to turn this model into a new one that solves all the three pain points in "our problem." Can you help us? The new model should retain most of the accuracy of the original one while also being faster, smaller, and in a supported format (extra points for a CoreML output).


## Rules and clarifications

1) Your submission should include a model we can use to test against the Fashion MNIST dataset, the code you used to generate it (a link to Google Colab would be great, so that we don't need to worry about installing the libraries you used 😽) and a brief summary of the tradeoff between model accuracy and model size (either in the notebook provided or in a separate pdf/markdown).
2) Creating just a smaller model might be a good starting point (i.e. simply a smaller topology), but keep in mind that we are interested in smarter approaches. After all, this is a test 👻
3) There are several methods to shrink a neural network while retaining most of its learning. If you choose to use one (or more), please also share with us some insights on the tradeoff introduced between parameters/output model size on disk and the model accuracy.


## Get inspired

Check out these great resources to get started:
1) https://www.tensorflow.org/model_optimization/guide/
2) https://keras.io/examples/vision/knowledge_distillation/
3) https://github.com/apple/coremltools


## I am done - but the Git Master branch is protected!

💻 As per git best practices, work on a new branch, commit your code there and when you are done perform a merge request onto master and send us an email!

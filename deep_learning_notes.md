# Deep Learning Notes

Note from [Ryan Shrott](https://towardsdatascience.com/@ryanshrott?source=post_header_lockup).


## Lesson 1: Why Deep Learning is taking off?

90% of all data was collected in the past 2 years. Deep neural networks (DNN’s) are capable of taking advantage of a very large amount of data. As a result, DNN’s can dominate smaller networks and traditional learning algorithms.

Furthermore, there have been a number of algorithmic innovations which have allowed DNN’s to train much faster. For example, switching from a sigmoid activation function to a RELU activation function has had a massive impact on optimization procedures such as gradient descent. These algorithmic improvements have allowed researchers to iterate throughout the IDEA -> EXPERIMENT -> CODE cycle much more quickly, leading to even more innovation.

## Lesson 2: Vectorization in Deep Learning

Before taking this course, I was not aware that a neural network could be implemented without any explicit for loops (except over the layers). Ng does an excellent job at conveying the importance of a vectorized code design in Python. The homework assignments provide you with a boilerplate vectorized code design which you could easily transfer to your own application.

## Lesson 3: Deep Understanding of DNN’s

The first course actually gets you to implement the forward and backward propagation steps in numpy from scratch. By doing this, I have gained a much deeper understanding of the inner workings of higher level frameworks such as TensorFlow and Keras. Ng explains the idea behind a computation graph which has allowed me to understand how TensorFlow seems to perform “magical optimization”.

## Lesson 4: Why Deep Representations?

Ng gives an intuitive understanding of the layering aspect of DNN’s. For example, in face detection he explains that earlier layers are used to group together edges in the face and then later layers use these edges to form parts of faces (i.e. nose, eyes, mouth etc.) and then further layers are used to put the parts together and identify the person. He also explains the idea of circuit theory which basically says that there exists functions which would require an exponential number of hidden units to fit the data in a shallow network. The exponential problem could be alleviated simply by adding a finite number of additional layers.

## Lesson 5: Tools for addressing Bias and Variance

Ng explains the steps a researcher would take to identify and fix issues related to bias and variance problems. The picture he draws gives a systematic approach to addressing these issues.

He also addresses the commonly quoted “tradeoff” between bias and variance. He explains that in the modern deep learning era we have tools to address each problem separately so that the tradeoff no longer exists.

## Lesson 6: Intuition for Regularization

Why does a penalization term added to the cost function reduce variance effects? The intuition I had before taking the course was that it forced the weight matrices to be closer to zero producing a more “linear” function. Ng gave another interpretation involving the tanh activation function. The idea is that smaller weight matrices produce smaller outputs which centralizes the outputs around the linear section of the tanh function.

He also gave an interesting intuitive explanation for dropout. Prior to taking the course I thought that dropout is basically killing random neurons on each iteration so it’s as if we are working with a smaller network, which is more linear. His intuition is to look at life from the perspective of a single neuron.

Since dropout is randomly killing connections, the neuron is incentivized to spread it’s weights out more evenly among its parents. By spreading out the weights, it tends to have the effect of shrinking the squared norm of the weights. He also explains that dropout is nothing more than an adaptive form of L2 regularization and that both methods have similar effects.

## Lesson 7: Why normalization works?

Ng demonstrates why normalization tends to improve the speed of the optimization procedure by drawing contour plots. He explicitly goes through an example of iterating through a gradient descent example on a normalized and non-normalized contour plot.

## Lesson 8: The importance of initialization

Ng shows that poor initialization of parameters can lead to vanishing or exploding gradients. He demonstrates several procedure to combat these issues. The basic idea is to ensure that each layer’s weight matrices has a variance of approximately 1. He also discusses Xavier initialization for tanh activation function.

## Lesson 9: Why mini-batch gradient descent is used?

Using contour plots, Ng explains the tradeoff between smaller and larger mini-batch sizes. The basic idea is that a larger size becomes to slow per iteration, while a smaller size allows you to make progress faster but cannot make the same guarantees regarding convergence. The best approach is do something in between which allows you to make progress faster than processing the whole dataset at once, while also taking advantage of vectorization techniques.

## Lesson 10: Intuitive understanding of advanced optimization techniques

Ng explains how techniques such as momentum and RMSprop allow gradient descent to dampen it’s path toward the minimum. He also gives an excellent physical explanation of the process with a ball rolling down a hill. He ties the methods together to explain the famous Adam optimization procedure.

## Lesson 11: Basic backend TensorFlow understanding

Ng explains how to implement a neural network using TensorFlow and also explains some of the backend procedures which are used in the optimization procedure. One of the homework exercises encourages you to implement dropout and L2 regularization using TensorFlow. This further strengthened my understanding of the backend processes.

## Lesson 12: Orthogonalization

Ng discusses the importance of orthogonalization in machine learning strategy. The basic idea is that you would like to implement controls that only affect a single component of your algorithms performance at a time. For example, to address bias problems you could use a bigger network or more robust optimization techniques. You would like these controls to only affect bias and not other issues such as poor generalization. An example of a control which lacks orthogonalization is stopping your optimization procedure early (early stopping). This is because it simultaneously affects the bias and variance of your model.

## Lesson 13: Importance of a single number evaluation metric

Ng stresses the importance of choosing a single number evaluation metric to evaluate your algorithm. You should only change the evaluation metric later on in the model development process if your target changes. Ng gives an example of identifying pornographic photos in a cat classification application!

## Lesson 14: Test/dev distributions

Always ensure that the dev and test sets have the same distribution. This ensures that your team is aiming at the correct target during the iteration process. This also means that if you decide to correct mislabeled data in your test set then you must also correct the mislabelled data in your development set.

## Lesson 15: Dealing with different training and test/dev distributions

Ng gives reasons for why a team would be interested in not having the same distribution for the train and test/dev sets. The idea is that you want the evaluation metric to be computed on examples that you actually care about. For example, you may want to use examples that are not as relevant to your problem for training, but you would not want your algorithm to be evaluated against these examples. This allows your algorithm to be trained with much more data. It has been empirically shown that this approach will give you better performance in many cases. The downside is that you have different distributions for your train and test/dev sets. The solution is to leave out a small piece of your training set and determine the generalization capabilities of the training set alone. Then you could compare this error rate to the actual development error and compute a “data mismatch” metric. Ng then explains methods of addressing this data mismatch problem such as artificial data synthesis.

## Lesson 16: Train/dev/test sizes

The guidelines for setting up the split of train/dev/test has changed dramatically during the deep learning era. Before taking the course, I was aware of the usual 60/20/20 split. Ng stresses that for a very large dataset, you should be using a split of about 98/1/1 or even 99/0.5/0.5. This is due to the fact that the dev and test sets only need to be large enough to ensure the confidence intervals provided by your team. If you are working with 10,000,000 training examples, then perhaps 100,000 examples (or 1% of the data) is large enough to guarantee certain confidence bounds on your dev and/or test set.

## Lesson 17: Approximating Bayes optimal error

Ng explains how human level performance could be used as a proxy for Bayes error in some applications. For example, for tasks such as vision and audio recognition, human level error would be very close to Bayes error. This allows your team to quantify the amount of avoidable bias your model has. Without a benchmark such as Bayes error, it’s difficult to understand the variance and avoidable bias problems in your network.

## Lesson 18: Error Analysis

Ng shows a somewhat obvious technique to dramatically increase the effectiveness of your algorithms performance using error analysis. The basic idea is to manually label your misclassified examples and to focus your efforts on the error which contributes the most to your misclassified data.

For example, in the cat recognition Ng determines that blurry images contribute the most to errors. This sensitivity analysis allows you see how much your efforts are worth on reducing the total error. It may be the case that fixing blurry images is an extremely demanding task, while other errors are obvious and easy to fix. Both the sensitivity and approximate work would be factored into the decision making process.

## Lesson 19: When to use transfer learning?

Transfer learning allows you to transfer knowledge from one model to another. For example, you could transfer image recognition knowledge from a cat recognition app to a radiology diagnosis. Implementing transfer learning involves retraining the last few layers of the network used for a similar application domain with much more data. The idea is that hidden units earlier in the network have a much broader application which is usually not specific to the exact task that you are using the network for. In summary, transfer learning works when both tasks have the same input features and when the task you are trying to learn from has much more data than the task you are trying to train.

## Lesson 20: When to use multi-task learning?

Multi-task learning forces a single neural network to learn multiple tasks at the same time (as opposed to having a separate neural network for each task). Ng explains that the approach works well when the set of tasks could benefit from having shared lower-level features and when the amount of data you have for each task is similar in magnitude.

## Lesson 21: When to use end-to-end deep learning?

End-to-end deep learning takes multiple stages of processing and combines them into a single neural network. This allows the data to speak for itself without the bias displayed by humans in hand engineering steps in the optimization procedure. To the contrary, this approach needs much more data and may exclude potentially hand designed components.

####################################################################################################

## Lesson 1: Why computer vision is taking off?

Big data and algorithmic developments will cause the testing error of intelligent systems to converge to Bayes optimal error. This will lead to AI surpassing human level performance in all areas, including natural perception tasks. Open source software from TensorFlow allows you to use transfer learning to implement an object detection system for any object very rapidly. With transfer learning, you only need about 100–500 examples for the system to work relatively well. Manually labeling 100 examples isn’t too much work, so you’ll have a minimum viable product very quickly.

## Lesson 2: How convolution works?

Ng explains how to implement the convolution operator and shows how it can detect edges in an image. He also explains other filters, such as the Sobel filter, which put more weight on central pixels of the edge. Ng then explains that the weights of the filter should not be hand-designed but rather should be learned using a hill climbing algorithm such as gradient descent.

## Lesson 3: Why convolutions?
Ng gives several philosophical reasons for why convolutions work so well in image recognition tasks. He outlines 2 concrete reasons. The first is known as parameter sharing. It is the idea that a feature detector that’s useful in one part of an image is probably useful in another part of the image. For example, an edge detector is probably useful is many parts of the image. The sharing of parameters allows the number of parameters to be small and also allows for robust translation invariance. Translation invariance is the notion that a cat shifted and rotated is still a picture of a cat.

The second idea he outlines is known as sparsity of connections. This is the idea that each output layer is only a function of a small number of inputs (particularly, the filter size squared). This greatly reduces the number of parameters in the network and allows for faster training.

## Lesson 3: Why Padding?

Padding is usually used to preserve the input size (i.e. the dimension of the input and output are the same). It is also used so that frames near the edges of image contribute as much to the output as frames near near the centre.

## Lesson 4: Why Max Pooling?

Through empirical research, max pooling has proven to be extremely effective in CNN’s. By downsampling the image, we reduce the number of parameters which makes the features invariant to scale or orientation changes.

## Lesson 5: Classical network architectures

Ng shows 3 classical network architectures including LeNet-5, AlexNet and VGG-16. The main idea he presents is that effective networks often have layers with an increasing channel size and decreasing width and height.

## Lesson 6: Why ResNets works?

For a plain network, the training error does not monotonically decrease as the number of layers increases due to vanishing and exploding gradients. These networks have feed forward skipped connections which allow you train extremely large networks without a drop in performance.

## Lesson 7: Use Transfer Learning!

Training large networks, such as inception, from scratch can take weeks on a GPU. You should download the weights from a pretrained network and just retrain the last softmax layer (or the last few layers). This will greatly reduce training time. The reason this works is that earlier layers tend to be associated with concepts in all images such as edges and curvy lines.

## Lesson 8: How to win computer vision competitions

Ng explains that you should train several networks independently and average their outputs to get better performance. Data augmentation techniques such as randomly cropping images, flipping images about the horizontal and vertical axes may also help with performance. Finally, you should use an open source implementation and pretrained model to start and then fine-tune the parameters for your particular application.

## Lesson 9: How to implement object detection

Ng starts by explaining the idea of landmark detection in an image. Basically, these landmarks become apart of your training output examples. With some clever convolution manipulations, you get an output volume that tells you the probability that the object is in a certain region and the location of the object. He also explains how to evaluate the effectiveness of your object detection algorithm using the intersection over union formula. Finally, Ng puts all these components together to explain the famous YOLO algorithm.

## Lesson 10: How to implement Face Recognition

Facial recognition is a one-shot learning problem since you may only have one example image to identify the person. The solution is to learn a similarity function which gives the degree of difference between two images. So if the images are of the same person, you want the function to output a small number, and vice versa for different people.

The first solution Ng gives is known as a siamese network. The idea is to input two persons into the same network separately and then compare their outputs. If the outputs are similar, then the persons are probably the same. The network is trained so that if two input images are of the same person, then the distance between their encodings is relatively small.

The second solution he gives uses a triplet loss method. The idea is that you have a triplet of images (Anchor (A), Positive (P) and Negative (N)) and you train the network so that the output distance between A and P is much smaller than the distance between A and N.

## Lesson 11: How to create artwork using Neural Style Transfer

Ng explains how to generate an image with a combining content and style. See the examples below.

The key to Neural Style Transfer is to understand the visual representations for what each layer in a convolutional network is learning. It turns out that earlier layers learn simple features like edges and later features learn complex objects like faces, feet and cars.

To build a neural style transfer image, you simply define a cost function which is a convex combination of the similarity in content and style. In particular, the cost function would be:

J(G) = alpha * J_content(C,G) + beta * J_style(S,G)

where G is the generated image, C is the content image and S is the style image. The learning algorithm simply uses gradient descent to minimize the cost function with respect to the generated image, G.

The steps are as follows:

1. Generate G randomly.

2. Use gradient descent to minimize J(G), i.e. write G := G-dG(J(G)).

3. Repeat step 2.



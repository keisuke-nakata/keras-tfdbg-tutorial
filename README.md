# keras-tfdbg-tutorial

A debugging sample for a broken keras program which tries to learns MNIST.

This is heavily based on https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/debug/examples/debug_mnist.py
and https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py .
The neural network in this demo has a problem with numerical computation (Infs and NaNs).

`python debug_minst.py --debug` triggers tfdbg interpreter.  
Run the program until inf or nans occur: `run -f has_inf_or_nan`.  
After some `run` commands (for filling value of inf or nans into cross_entropy tensors), you can see tensors with inf or nans: `lt -f has_inf_or_nan`.  
You will see -inf values on Log tensor (`pt Log:0`) and 0 values on Softmax tensor (`pt dense_2/Softmax:0`).  
This shows that our defined cross_entropy function (`def unstable_categorical_crossentropy`) is broken and why our accuracy goes to nan.  
To fix the problem, comment-out the line 74 (loss as our function) and use line 75 (loss as keras defined function) instead.

For more detail, see:
- [Debugging TensorFlow Programs  |  TensorFlow](https://www.tensorflow.org/programmers_guide/debugger)
- How to use tfdbg in Keras (Japanese article): https://qiita.com/keisuke-nakata/items/5d6918678e8099b565d0

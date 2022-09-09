# Stock-Market-Anomaly-Detection
Time Series Anomaly Detection with LSTM Autoencoders using Keras in Python

This guide will show you how to build an Anomaly Detection model for Time Series data. You’ll learn how to use LSTMs and Autoencoders in Keras and TensorFlow 2. We’ll use the model to find anomalies in S&P 500 daily closing prices.

This is the plan:

Anomaly Detection
LSTM Autoencoders
S&P 500 Index Data
LSTM Autoencoder in Keras
Finding Anomalies

# Anomaly Detection
Anomaly detection refers to the task of finding/identifying rare events/data points. Some applications include - bank fraud detection, tumor detection in medical imaging, and errors in written text.

A lot of supervised and unsupervised approaches to anomaly detection has been proposed. Some of the approaches include - One-class SVMs, Bayesian Networks, Cluster analysis, and (of course) Neural Networks.

We will use an LSTM Autoencoder Neural Network to detect/predict anomalies (sudden price changes) in the S&P 500 index.

# LSTM Autoencoders
Autoencoders Neural Networks try to learn data representation of its input. So the input of the Autoencoder is the same as the output? Not quite. Usually, we want to learn an efficient encoding that uses fewer parameters/memory.

The encoding should allow for output similar to the original input. In a sense, we’re forcing the model to learn the most important features of the data using as few parameters as possible.

# Anomaly Detection with Autoencoders
Here are the basic steps to Anomaly Detection using an Autoencoder:

1- Train an Autoencoder on normal data (no anomalies)
2- Take a new data point and try to reconstruct it using the Autoencoder
3- If the error (reconstruction error) for the new data point is above some threshold, we label the example as an anomaly

Good, but is this useful for Time Series Data? Yes, we need to take into account the temporal properties of the data. Luckily, LSTMs can help us with that.

# S&P 500 Index Data

Our data is the daily closing prices for the S&P 500 index from 1986 to 2020.

- The S&P 500, or just the S&P, is a stock market index that measures the stock performance of 500 large companies listed on stock exchanges in the United States. It - - is one of the most commonly followed equity indices, and many consider it to be one of the best representations of the U.S. stock market. -Wikipedia

It is provided by Patrick David and hosted on Kaggle. The data contains only two columns/features - the date and the closing price. Let’s download and load into a Data Frame:

# Preprocessing
We’ll use 80% of the data and train our model on it:
Next, we’ll rescale the data using the training data and apply the same transformation to the test data:
Finally, we’ll split the data into subsequences. Here’s the little helper function for that:
We’ll create sequences with 30 days worth of historical data:
The shape of the data looks correct. How can we make LSTM Autoencoder in Keras?

# LSTM Autoencoder in Keras
Our Autoencoder should take a sequence as input and outputs a sequence of the same shape. Here’s how to build such a simple model in Keras:
There are a couple of things that might be new to you in this model. The RepeatVector layer simply repeats the input n times. Adding return_sequences=True in LSTM layer makes it return the sequence.

Finally, the TimeDistributed layer creates a vector with a length of the number of outputs from the previous layer. Your first LSTM Autoencoder is ready for training.

Training the model is no different from a regular LSTM model:

# Evaluation
We’ve trained our model for 10 epochs with less than 8k examples. Here are the results:

# Finding Anomalies
Still, we need to detect anomalies. Let’s start with calculating the Mean Absolute Error (MAE) on the training data:

We’ll pick a threshold of 0.65, as not much of the loss is larger than that. When the error is larger than that, we’ll declare that example an anomaly:

Let’s calculate the MAE on the test data:

We’ll build a DataFrame containing the loss and the anomalies (values above the threshold):

Looks like we’re thresholding extreme values quite well. Let’s create a DataFrame using only those:

Finally, let’s look at the anomalies found in the testing data:

You should have a thorough look at the chart. The red dots (anomalies) are covering most of the points with abrupt changes to the closing price. You can play around with the threshold and try to get even better results.

# Conclusion
You just combined two powerful concepts in Deep Learning - LSTMs and Autoencoders. The result is a model that can find anomalies in S&P 500 closing price data. You can try to tune the model and/or the threshold to get even better results.

Here’s a recap of what you did:

1- Anomaly Detection
2- LSTM Autoencoders
3- S&P 500 Index Data
4- LSTM Autoencoder in Keras
5- Finding Anomalies

# References
1- https://www.tensorflow.org/tutorials/structured_data/time_series
2- https://colah.github.io/posts/2015-08-Understanding-LSTMs/
3- https://towardsdatascience.com/step-by-step-understanding-lstm-autoencoder-layers-ffab055b6352
4- https://www.kaggle.com/datasets/pdquant/sp500-daily-19862018

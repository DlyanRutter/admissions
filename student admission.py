import pandas as pd
import numpy as np
import keras, math
import tensorflow as tf

df=pd.read_csv('/Users/dylanrutter/Downloads/aind2-dl-master/student_data.csv')
df=df.fillna(0) #fill NA slots with 0
df = pd.get_dummies(df, columns=['rank'])

df['gre'] = df['gre']/800 #normalize from range 0 to 800
df['gpa'] = df['gpa']/4

X = np.array(df)[:,1:]#makes array where each element is a vector of row
X = X.astype('float32')
y = keras.utils.np_utils.to_categorical(df['admit'], 2)

X_train, X_test = X[50:], X[:50]
y_train, y_test = y[50:], y[:50]


def make_logits(X1, y=2, h1_nodes=252, h2_nodes=128, h3_nodes=45):
    """
    X is an input array of features with shape (num_samples, num_features),
    y is an input array of labels with shape (num_samples, num_labels),
    h1_nodes is the number of nodes in hidden layer 1, h2_nodes is the number
    of nodes in hidden layer 2, h3_nodes is the number of nodes in hidden
    layer 3. y is the number of labels Returns neural network logits.
    """

    weights1 = tf.Variable(
        tf.truncated_normal([X1.shape[1], h1_nodes],
                             stddev=1.0 / math.sqrt(float(X1.shape[1]))))
    biases1 = tf.Variable(tf.zeros([h1_nodes]))
    hl1_net = tf.add(tf.matmul(X1, weights1), biases1)
    hl1_out = tf.nn.relu(hl1_net)

    weights2 = tf.Variable(
        tf.truncated_normal([h1_nodes, h2_nodes],
                            stddev = 1.0 / math.sqrt(float(h1_nodes))))
    biases2 = tf.Variable(tf.zeros([h2_nodes]))
    h2_net = tf.add(tf.matmul(hl1_out, weights2), biases2)
    h2_out = tf.nn.relu(h2_net)

    weights3 = tf.Variable(
        tf.truncated_normal([h2_nodes, h3_nodes],
                            stddev=1.0 / math.sqrt(float(h2_nodes))))
    biases3 = tf.Variable(tf.zeros([h3_nodes]))
    h3_net = tf.add(tf.matmul(h2_out, weights3), biases3)
    h3_out = tf.nn.relu(h3_net)

    output_weights = tf.Variable(
        tf.truncated_normal([h3_nodes, y],
                            stddev = 1.0 / math.sqrt(float(h3_nodes))))
    output_biases = tf.Variable(tf.zeros([y]))
    output_logits = tf.matmul(h3_out, output_weights) + output_biases

    return output_logits

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

batch_size = 12
num_steps = 3001
graph = tf.Graph()

with graph.as_default():

    features_placeholder = tf.placeholder(tf.float32,
                                         shape=(batch_size, X.shape[1]))
    labels_placeholder = tf.placeholder(tf.float32,
                                       shape=(None, y.shape[1]))
    keep_prob = tf.placeholder(tf.float32)

    logits = make_logits(X_train, y_train) ###problem is here need to do
        ###only for placeholders
    print logits.get_shape()
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=labels_placeholder, logits=logits))
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    train_pred = tf.nn.softmax(logits)
    test_pred = tf.nn.softmax(make_logits(X_test, y_test))
    
 #   correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
#    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()

    for step in range(num_steps):
        
        offset = (step*batch_size) % (y_train.shape[0] - batch_size)
        batch_data = X_train[offset:(offset + batch_size), :]
        batch_labels = y_train[offset:(offset + batch_size), :]

        feed_dict = {features_placeholder:batch_data,
                     labels_placeholder:batch_labels}

        _, log_loss, predictions = sess.run(
            [optimizer, loss, train_pred], feed_dict=feed_dict)
        if (step % 100 == 0):
            print "loss at step %d: %f" % (step,1)
            print "minibatch acc: %.1f%%"  % accuracy(predictions, batch_labels)
    print("Test accuracy: %.1f%%" % accuracy(test_pred.eval(), test_labels))



"""
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
weights = weights.eval(sess)
print weights
"""

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

#makes labels into one-hot array
y = df['admit'].values.ravel()
num_classes = np.max(y) + 1
n = y.shape[0]
categorical = np.zeros((n, num_classes))
categorical[np.arange(n),y] = 1
y = categorical

#splits data into train and test sets of features and labels
X_train, X_test = X[50:], X[:50]
y_train, y_test = y[50:], y[:50]

#batch size and number of epochs
batch_size = 12
num_steps = 3001

#number of nodes in each hidden layer
h1_nodes = 128
h2_nodes = 64
h3_nodes = 32

#function for checking accuracy
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

#function for making logits
def make_logits(X_train, X_test, Train=True):
    
    #set up weights, biases, output for first layer, if training on train set,
    #use shape of X_train, X_test shape otherwise
    weights1 = tf.Variable(
        tf.truncated_normal([X.shape[1], h1_nodes],
                        stddev=1.0 / math.sqrt(float(X.shape[1]))))
    biases1 = tf.Variable(tf.zeros([h1_nodes]))
    if Train == True:
        hl1_net = tf.add(tf.matmul(X_train, weights1), biases1)
    else:
        hl1_net = tf.add(tf.matmul(X_test, weights1), biases1)
    hl1_out = tf.nn.relu(hl1_net)
    
    #set up weights, biases, output for second layer
    weights2 = tf.Variable(
        tf.truncated_normal([h1_nodes, h2_nodes],
                            stddev = 1.0 / math.sqrt(float(h1_nodes))))
    biases2 = tf.Variable(tf.constant(1.0, shape=[h2_nodes]))
    h2_net = tf.add(tf.matmul(hl1_out, weights2), biases2)
    h2_out = tf.nn.relu(h2_net)

    #set up weights, biases, output for third layer
    weights3 = tf.Variable(
        tf.truncated_normal([h2_nodes, h3_nodes],
                            stddev=1.0 / math.sqrt(float(h2_nodes))))
    biases3 = tf.Variable(tf.constant(1.0, shape=[h3_nodes]))
    h3_net = tf.add(tf.matmul(h2_out, weights3), biases3)
    h3_out = tf.nn.relu(h3_net)
    
    #set up weights, biases, output for output layer
    output_weights = tf.Variable(
        tf.truncated_normal([h3_nodes, y.shape[1]],
                            stddev = 1.0 / math.sqrt(float(X.shape[1]))))
    output_biases = tf.Variable(tf.constant(1.0, shape=[y.shape[1]]))
    logits = tf.matmul(h3_out, output_weights) + output_biases
    return logits

graph = tf.Graph()
with graph.as_default():
    #set up placeholders and constants
    X_train_tf = tf.placeholder(tf.float32,
                                shape=(batch_size, X.shape[1]))
    y_train_tf = tf.placeholder(tf.float32,
                                shape=(batch_size, y.shape[1]))
    X_test_tf = tf.constant(X_test)
    y_test_tf = tf.constant(y_test)
    
    #set up logits, loss, optimizer
    logits = make_logits(X_train_tf, X_test_tf, Train=True)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=y_train_tf, logits=logits))
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    #make train and test preds
    train_pred = tf.nn.softmax(logits)
    test_pred = tf.nn.softmax(make_logits(X_train_tf, X_test_tf, Train=False))
    
with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()

    for step in range(num_steps):
        #set up a batch generator and feed dict        
        offset = (step*batch_size) % (y_train.shape[0] - batch_size)
        batch_data = X_train[offset:(offset + batch_size), :]
        batch_labels = y_train[offset:(offset + batch_size), :]
        feed_dict = {X_train_tf:batch_data,
                     y_train_tf:batch_labels}
        
        #run session, printing log loss and minibatch accuracy
        _, log_loss, predictions = sess.run(
            [optimizer, loss, train_pred], feed_dict=feed_dict)
        if (step % 100 == 0):
            print "loss at step %d: %f" % (step,1)
            print "minibatch acc: %.1f%%"  % accuracy(predictions, batch_labels)
    print("Test accuracy: %.1f%%" % accuracy(test_pred.eval(), y_test))
    
hl1 = {'weights': tf.Variable(tf.truncated_normal([num_features, h1_nodes])),
       'biases' : tf.Variable(tf.zeros([h1_nodes]))}
hl2 = {'weights': tf.Variable(tf.truncated_normal([h1_nodes, h2_nodes])),
       'biases' : tf.Variable(tf.zeros([h2_nodes]))}
hl3 = {'weights': tf.Variable(tf.truncated_normal([h2_nodes, h3_nodes])),
       'biases' : tf.Variable(tf.zeros([h3_nodes]))}
out = {'weights': tf.Variable(tf.truncated_normal([h3_nodes, num_labels])),
       'biases' : tf.Variable(tf.zeros([num_labels]))}

keep_prob = tf.placeholder(tf.float32)
lg1 = tf.add(tf.matmul(X_train, hl1['weights']), hl1['biases'])
lg1 = tf.nn.relu(lg1)
lg1 = tf.nn.dropout(lg1, keep_prob)

lg2 = tf.add(tf.matmul(lg1, hl2["weights"]), hl2['biases'])
lg2 = tf.nn.relu(lg2)
lg2 = tf.nn.dropout(lg2, keep_prob)

lg3 = tf.add(tf.matmul(lg2, hl3['weights']), hl3['biases'])
lg3 = tf.nn.relu(lg3)
lg3 = tf.nn.dropout(lg3, keep_prob)

logits = tf.matmul(lg3, out['weights']) + out['biases']

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=y_train))
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

tf_train_features = tf.placeholder(tf.float32, shape=(batch_size, num_features))
tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
tf_valid_features = tf.constant(X_valid)
tf_test_features = tf.constant(X_test)
                      
train_pred = tf.nn.softmax(logits)
valid_pred = tf.nn.softmax(tf.matmul(X_valid, out['weights']), out['biases'])
test_pred = tf.nn.softmax(tf.matmul(X_test, out['weights']), out['biases'])

num_steps = 3001
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for step in range(num_steps):
        #set up a batch generator and feed dict        
        offset = (step*batch_size) % (y_train.shape[0] - batch_size)
        batch_data = X_train[offset:(offset + batch_size), :]
        batch_labels = y_train[offset:(offset + batch_size), :]
        feed_dict = {tf_train_features:batch_data,
                     tf_train_labels:batch_labels}
        
        #run session, printing log loss and minibatch accuracy
        _, log_loss, predictions = sess.run(
            [optimizer, loss, train_pred], feed_dict=feed_dict)
        if (step % 100 == 0):
            print "loss at step %d: %f" % (step,1)
            print "minibatch acc: %.1f%%"  % accuracy(predictions, batch_labels)
    print("Test accuracy: %.1f%%" % accuracy(test_pred.eval(), y_test))


import pandas as pd
import numpy as np
import keras, math
import tensorflow as tf

def accuracy(predictions, labels):
    """
    determines accuracy
    """
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

#load dataset
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

#splits data into train, valid, and test sets of features and labels
X_train, X_valid, X_test = X[:900], X[900:1050], X[1050:1200]
y_train, y_valid, y_test = y[:900], y[900:1050], y[1050:1200]

#set up univeral variables
num_features = X_train.shape[1]
num_labels = y_train.shape[1]
batch_size = 100
epochs = 3002
learn_rate = 0.01
num_steps = 3001

#number of nodes in each hidden layer
h1_nodes = 528
h2_nodes = 256
h3_nodes = 128

graph = tf.Graph()
with graph.as_default():

    #make placeholders for batches
    tf_train_features = tf.placeholder(tf.float32,
                                       shape=(batch_size, num_features))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_features = tf.constant(X_valid)
    tf_test_features = tf.constant(X_test)

    #set up weights and biases
    initializer = tf.contrib.layers.xavier_initializer()
    hl1 = {'weights': tf.Variable(initializer([num_features, h1_nodes])),
           'biases' : tf.Variable(tf.zeros([h1_nodes]))}
    hl2 = {'weights': tf.Variable(initializer([h1_nodes, h2_nodes])),
           'biases' : tf.Variable(tf.zeros([h2_nodes]))}
    hl3 = {'weights': tf.Variable(initializer([h2_nodes, h3_nodes])),
           'biases' : tf.Variable(tf.zeros([h3_nodes]))}
    out = {'weights': tf.Variable(initializer([h3_nodes, num_labels])),
           'biases' : tf.Variable(tf.zeros([num_labels]))}
    
    #establish a keep_probability placeholder for logits and compute logits
    #for each layer
    keep_prob = tf.placeholder(tf.float32)
    lg1 = tf.add(tf.matmul(tf_train_features, hl1['weights']), hl1['biases'])
    lg1 = tf.nn.relu(lg1)
    lg1 = tf.nn.dropout(lg1, keep_prob)

    lg2 = tf.add(tf.matmul(lg1, hl2["weights"]), hl2['biases'])
    lg2 = tf.nn.relu(lg2)
    lg2 = tf.nn.dropout(lg2, keep_prob)

    lg3 = tf.add(tf.matmul(lg2, hl3['weights']), hl3['biases'])
    lg3 = tf.nn.relu(lg3)
    lg3 = tf.nn.dropout(lg3, keep_prob)
    logits = tf.matmul(lg3, out['weights']) + out['biases']
    
    #compute loss function of cross entropy + regularization
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=tf_train_labels)) +\
        0.01*tf.nn.l2_loss(hl1['weights']) +\
        0.01* tf.nn.l2_loss(hl2['weights']) + 0.01*tf.nn.l2_loss(
            hl3['weights']) + 0.01*tf.nn.l2_loss(out['weights'])

    #make an optimizer with a decaying learning rate               
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.5, global_step, 10000, 0.96)
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate).minimize(loss)
    
    #make logits for validation set
    Vlg1 = tf.add(tf.matmul(tf_valid_features, hl1['weights']), hl1['biases'])
    Vlg1 = tf.nn.relu(Vlg1)

    Vlg2 = tf.add(tf.matmul(Vlg1, hl2["weights"]), hl2['biases'])
    Vlg2 = tf.nn.relu(Vlg2)

    Vlg3 = tf.add(tf.matmul(Vlg2, hl3['weights']), hl3['biases'])
    Vlg3 = tf.nn.relu(Vlg3)
    Vlogits = tf.matmul(Vlg3, out['weights']) + out['biases']
    
    #make logits for test set
    Tlg1 = tf.add(tf.matmul(tf_test_features, hl1['weights']), hl1['biases'])
    Tlg1 = tf.nn.relu(Tlg1)

    Tlg2 = tf.add(tf.matmul(Tlg1, hl2["weights"]), hl2['biases'])
    Tlg2 = tf.nn.relu(Tlg2)

    Tlg3 = tf.add(tf.matmul(Tlg2, hl3['weights']), hl3['biases'])
    Tlg3 = tf.nn.relu(Tlg3)
    Tlogits = tf.matmul(Tlg3, out['weights']) + out['biases']
    
    #set up predictions/create probabilities
    train_pred = tf.nn.softmax(logits)
    valid_pred = tf.nn.softmax(Vlogits)
    test_pred = tf.nn.softmax(Tlogits)

#initialize session
with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()

    #run step number of epochs
    for step in range(epochs):

        #establish batches
        offset = (step * batch_size) % (y_train.shape[0] - batch_size)
        batch_features = X_train[offset:(offset + batch_size), :]
        batch_labels = y_train[offset:(offset + batch_size), :]

        #create feed dict mapping placeholders to batches, learning rate to
        #learning rate, and keep probability to keep probability
        train_feed_dict = {tf_train_features:batch_features,
                           tf_train_labels:batch_labels,
                           learning_rate:learn_rate,
                           keep_prob:0.5}

        #run session to find log loss and predictions
        _, l, predictions = sess.run(
            [optimizer, loss, train_pred], feed_dict=train_feed_dict)
        
        #Get stats
        if (step % 500 == 0):
            print("Minibatch loss at step {}: {}".format(step, l))
            print("Minibatch accuracy: {:.1f}".format(
                accuracy(predictions, batch_labels)))
            print("Validation accuracy: {:.1f}".format(
                accuracy(valid_pred.eval(), y_valid)))
    print("Test accuracy: {:.1f}".format(accuracy(test_pred.eval(), y_test)))


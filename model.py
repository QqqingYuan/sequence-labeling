__author__ = 'PC-LiNing'

import tensorflow as tf
import data_helpers
import datetime
import numpy
import load_data
import argparse


vocab_size = 28782
embedding_size = 200
max_sentence_length = 40
num_steps = max_sentence_length
num_hidden = 100
num_tag = 9
rnn_layer = 1

# training
Test_size = 3327
Train_size = 16551
BATCH_SIZE = 128
EVAL_FREQUENCY = 100
NUM_EPOCHS = 20

FLAGS = None


def train():
    # load data
    # vectors = [28782+1,200]
    # vectors = numpy.load('vectors.npy')
    train_sent,train_label,train_length,test_sent,test_label,test_length = load_data.load_train_test()
    # word embedding
    words_embedding = tf.random_uniform([vocab_size+1, embedding_size], -1.0, 1.0, name="embedding")
    # input is a sentence
    train_data_node = tf.placeholder(tf.int32, shape=(None,max_sentence_length))
    train_length_node = tf.placeholder(tf.int32, shape=(None,))
    train_labels_node = tf.placeholder(tf.int32, shape=(None, max_sentence_length))

    weights = tf.Variable(tf.random_uniform([num_hidden*2, num_tag], -1.0, 1.0), name="w")
    biases = tf.Variable(tf.random_normal(shape=[num_tag], dtype=tf.float32), name="b")
    # CRF
    transitions = tf.Variable(tf.random_uniform([num_tag, num_tag], -1.0, 1.0), name="trans")

    # inputs = [batch_size,max_sentence_length]
    # lengths = [batch_size,]
    def blstm_crf(inputs):
        # sents = [batch_size,max_path_length,embedding_size]
        sents = tf.nn.embedding_lookup(words_embedding, inputs)
        x = tf.transpose(sents, [1, 0, 2])
        x = tf.reshape(x, [-1, embedding_size])
        #  get a list of 'n_steps' tensors of shape (batch_size, embeddings)
        x = tf.split(0, num_steps, x)
        # bi-lstm
        fw_cell = tf.nn.rnn_cell.LSTMCell(num_hidden,forget_bias=1.0,state_is_tuple=True)
        fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=0.5)
        bw_cell = tf.nn.rnn_cell.LSTMCell(num_hidden,forget_bias=1.0,state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=0.5)
        if rnn_layer > 1:
            fw_cell = tf.nn.rnn_cell.MultiRNNCell([fw_cell] * rnn_layer)
            bw_cell = tf.nn.rnn_cell.MultiRNNCell([bw_cell] * rnn_layer)

        # output = [batch_size,num_hidden*2]
        outputs, fw_final_state, bw_final_state = tf.nn.bidirectional_rnn(fw_cell, bw_cell,x, dtype=tf.float32)
        # linear
        # rnn_output = [batch_size,num_steps,num_hidden*2]
        rnn_output = tf.transpose(tf.pack(outputs), perm=[1, 0, 2])
        # output = [batch_size*num_steps,num_tag]
        output = tf.matmul(tf.reshape(rnn_output, [-1, num_hidden*2]), weights) + biases
        # output = [batch_size,num_steps,num_tag]
        output = tf.reshape(output, [-1, num_steps, num_tag])
        return output
    # unary_scores = [batch_size,num_steps,num_tag]
    unary_scores = blstm_crf(train_data_node)
    # CRF
    log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(unary_scores,train_labels_node,train_length_node,transition_params=transitions)
    loss = tf.reduce_mean(-log_likelihood, name='cross_entropy_mean_loss')

    # train
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # runing the training
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        print('Initialized!')
        # generate batches
        batches = data_helpers.batch_iter(list(zip(train_sent,train_label,train_length)),BATCH_SIZE,NUM_EPOCHS)
        # batch count
        batch_count = 0
        epoch = 1
        print("Epoch "+str(epoch)+":")
        for batch in batches:
            batch_count += 1
            # train process
            x_batch, y_batch, length_batch = zip(*batch)
            feed_dict = {train_data_node: x_batch,train_labels_node: y_batch,train_length_node:length_batch}
            _,step,losses, tf_transition_params = sess.run([train_op, global_step,loss,transition_params],feed_dict=feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}".format(time_str, step, losses))
            # test process
            if float((batch_count * BATCH_SIZE) / Train_size) > epoch:
                epoch += 1
                print("Epoch "+str(epoch)+":")
            if batch_count % EVAL_FREQUENCY == 0:
                # get test scores
                feed_dict = {train_data_node: test_sent,train_labels_node: test_label,train_length_node:test_length}
                step,losses,scores = sess.run([global_step,loss,unary_scores],feed_dict=feed_dict)
                correct_labels = 0
                total_labels = 0
                for i in range(Test_size):
                    # Remove padding from the scores and tag sequence.
                    current_score = scores[i][:test_length[i]]
                    current_label = test_label[i][:test_length[i]]
                    # Compute the highest scoring sequence.
                    viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(current_score, tf_transition_params)
                    # Evaluate word-level accuracy.
                    correct_labels += numpy.sum(numpy.equal(viterbi_sequence, current_label))
                    total_labels += test_length[i]

                time_str = datetime.datetime.now().isoformat()
                acc = 100.0 * correct_labels / float(total_labels)
                print("\n")
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, losses, acc))
                print("\n")


def main(_):
    # if tf.gfile.Exists(FLAGS.summaries_dir):
    #    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    # tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summaries_dir', type=str, default='/tmp/blstm_crf',help='Summaries directory')
    FLAGS = parser.parse_args()
    tf.app.run()

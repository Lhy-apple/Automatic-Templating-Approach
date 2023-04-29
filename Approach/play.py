import tensorflow as tf
import numpy as np
import statistical_analysis as sa
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
from sklearn import model_selection as cv
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from sklearn.metrics import classification_report

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("frc", 1)
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("source", "Dataset_new", "source projects, split by space")


# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 192, "Dimensionality of character embedding")
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.2, "L2 regularizaion lambda (default: 0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs")
tf.flags.DEFINE_integer("evaluate_every", 1, "Evaluate model on dev set after this many epochs")
tf.flags.DEFINE_integer("checkpoint_every", 5000, "Save model after this many steps")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()


class_names=('describe','expected','reproduce','actual','environment','additional')



def train_model(allow_save_model = False, print_intermediate_results = True):

    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    # Data Preparatopn
    # ==================================================

    # Load data
    print("Loading data...")

    #load each source project
    project = FLAGS.source
    frc = FLAGS.frc
    source_text = np.array([])
    source_y = np.array([])
    target_text = np.array([])
    target_y = np.array([])

    source_file_path = "data/" + project + "/"
    target_file_path = "data/" + project + "/"
    fault_file = open(target_file_path+ 'fp_sentences_'+str(frc)+".txt", 'a')
    result_file = open(target_file_path + 'result_file.txt', 'a')

    source_files = list()
    for class_name in class_names:
        source_files.append(source_file_path + class_name)

    source_text, source_y,verify_text,verify_y,target_text,target_y = data_helpers.load_data_and_labels(source_files,frc)
    all_text = np.concatenate([source_text, target_text], 0)
    all_text = np.concatenate([all_text, verify_text], 0)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in all_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    source_x = np.array(list(vocab_processor.fit_transform(source_text)))
    target_x = np.array(list(vocab_processor.fit_transform(target_text)))
    verify_x = np.array(list(vocab_processor.fit_transform(verify_text)))
    print(source_text[0:2])
    print(source_x[0:2])

    if print_intermediate_results:
        print("\n*************frc", frc, "************", file=result_file)
        print('**data distribution in source dataset',file=result_file)
        sa.print_data_distribution(source_y, class_names,result_file)
        print('**data distribution in verify dataset',file=result_file)
        sa.print_data_distribution(verify_y, class_names,result_file)
        print('**data distribution in target dataset',file=result_file)
        sa.print_data_distribution(target_y, class_names,result_file)

        print("**Max Document Length: {:d}".format(max_document_length),file=result_file)
        print("**Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)),file=result_file)
        print("**Train/Test size: {:d}/{:d}".format(len(source_y), len(target_y)),file=result_file)

    # Training
    # ==================================================

    min_loss = 100000000
    predictions_at_min_loss = None
    steps_per_epoch = (int)(len(source_y) / FLAGS.batch_size) + 1

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=source_x.shape[1],
                num_classes=source_y.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)


            learning_rate = tf.train.polynomial_decay(2*1e-3, global_step,
                                                      steps_per_epoch * FLAGS.num_epochs, 1e-4,
                                                      power=1)

            optimizer = tf.train.AdamOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            if allow_save_model:

                # Output directory for models and summaries
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(
                    os.path.join(os.path.curdir, "runs", FLAGS.source))
                print("Writing to {}\n".format(out_dir))

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir_name = "checkpoint-" + str(FLAGS.embedding_dim) + "-" + FLAGS.filter_sizes + "-" + \
                                      str(FLAGS.num_filters) + "-" + str(FLAGS.dropout_keep_prob) + "-" + str(
                    FLAGS.l2_reg_lambda) + \
                                      "-" + str(FLAGS.batch_size) + "-" + str(FLAGS.num_epochs)
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, checkpoint_dir_name))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.all_variables())

                # Write vocabulary
                vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer(), feed_dict={cnn.phase_train: True})  # this is for version r0.12

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    cnn.phase_train: True
                }
                _, step, loss, mean_loss, l2_loss, accuracy = sess.run(
                    [train_op, global_step, cnn.loss, cnn.mean_loss, cnn.l2_loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                # print("{}: step {}, loss {:g}, acc {:g}, mean_loss {}, l2_loss {}".format(time_str, step, loss, accuracy, mean_loss, l2_loss))
                return accuracy

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0,
                    cnn.phase_train: False
                }
                step, loss, mean_loss, l2_loss, accuracy, predictions = sess.run(
                    [global_step, cnn.loss, cnn.mean_loss, cnn.l2_loss, cnn.accuracy, cnn.predictions],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()

                if print_intermediate_results:
                    print("{}: epoch {}, step {}, loss {:g}, acc {:g}, mean_loss {}, l2_loss {}".format(
                        time_str, step/steps_per_epoch, step, loss, accuracy, mean_loss, l2_loss))
                return accuracy, loss, predictions

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(source_x, source_y)), FLAGS.batch_size, FLAGS.num_epochs)

            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_accuracy = train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                current_epoch = current_step/steps_per_epoch
                if current_step%steps_per_epoch==0 and current_epoch % FLAGS.evaluate_every == 0:
                    if print_intermediate_results:
                        print("Current train accuracy: %s\nEvaluation:" % (train_accuracy))

                    fold_accuracy, loss, predictions = dev_step(verify_x, verify_y)
                    # ensemble_prediction([predictions], target_y)
                    if loss < min_loss:
                        min_loss = loss
                        predictions_at_min_loss = predictions
                        if allow_save_model:
                            save_path = saver.save(sess, checkpoint_prefix)
                            if print_intermediate_results:
                                print("Model saved in file: %s" % save_path)


            # Final result
            print('**Final result:',file = result_file)
            fold_accuracy, loss, predictions = dev_step(target_x, target_y)
            print("  ACC: %s" % (fold_accuracy),file = result_file)
            tp, fp, fn, precision, recall, f1 = sa.calculate_IR_metrics(target_text, target_y, predictions, class_names,
                                                                        fault_file)
            for i in range(len(class_names)):
                print("  ",class_names[i], precision[i], recall[i], f1[i],file = result_file)
            print("  average f1-score: %s" % (sum(f1) / len(f1)),file = result_file)
            result_file.close()
            fault_file.close()

    return min_loss, predictions_at_min_loss, target_y


train_model(False,True)





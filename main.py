from LoadData import LoadData
from RCF import RCF
from FM import FM
import numpy
import os, time
from tensorflow.app import flags
import tensorflow as tf

flags.DEFINE_string(
    "path", default="data/", help="data path")
flags.DEFINE_string(
    "dataset", default='frappe', help="Number of training steps to run.")
flags.DEFINE_integer(
    "feature_dim", default=20, help="dimension of the feature latent vectors")
flags.DEFINE_integer(
    "user_dim", default=20, help="dimension of the user latent vectors")
flags.DEFINE_string(
    "combine_method", default='sum', help="method for combining relational result (sum, attention)")
flags.DEFINE_float(
    "lr", default=0.0001, help="learning rate")
flags.DEFINE_integer(
    "iterations", default=10000, help="training epoch")
flags.DEFINE_integer(
    "batch_size", default=2048, help="batch size")


FLAGS = flags.FLAGS

def main():

    def evaluate_test_data():
        test_dict = data_loader.Test_data
        feed_dict = {model.users_ph: test_dict['U'], model.attr_ph: test_dict['X'], model.ratings_ph: test_dict['Y']}
        test_predict, test_rmse = sess.run([model.predict, model.rmse], feed_dict=feed_dict)
        print("test rmse: %.5f" %(test_rmse))
        print(test_predict)
    
    
    data_loader = LoadData(FLAGS.path, FLAGS.dataset, 'nolog_loss', FLAGS.batch_size)

    #model = RCF(FLAGS, data_loader.features_M , data_loader.users_M, data_loader.feature_n )
    model = FM(FLAGS, data_loader.features_M , data_loader.users_M, data_loader.feature_n )


    train_iters = data_loader.Train_iters

    print('start training...')
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(train_iters.initializer)
    train_data_n = data_loader.train_data_n
    index = 0
    for i in range(FLAGS.iterations):
        if index * FLAGS.batch_size > train_data_n:
            sess.run(train_iters.initializer)
            evaluate_test_data()
            index = 0
        (user_bt, attr_bt, rating_bt) = sess.run(train_iters.get_next())
        #bt means "batch training"
        feed_dict = {model.users_ph: user_bt, model.attr_ph: attr_bt, model.ratings_ph: rating_bt}
        _, loss, output = sess.run([model.update, model.loss, model.predict], feed_dict=feed_dict)

        index += 1

        if i % 10 == 0:
            print('iter %d, loss=%.5f' %(i, loss))



    

if __name__ == "__main__":
    main()    
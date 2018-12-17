import numpy as np
import os
import tensorflow as tf

class LoadData(object):
    '''given the path of data, return the data format for DeepFM
    :param path
    return:
    Train_data: a dictionary, 'Y' refers to a list of y values; 'X' refers to a list of features_M dimension vectors with 0 or 1 entries
    Test_data: same as Train_data
    Validation_data: same as Train_data
    '''

    # Three files are needed in the path
    def __init__(self, path, dataset, loss_type, batch_size):
        self.path = path + dataset + "/"
        self.trainfile = self.path + dataset +".train.libfm"
        self.testfile = self.path + dataset + ".test.libfm"
        self.validationfile = self.path + dataset + ".validation.libfm"
        self.features_M, self.users_M = self.map_features()
        self.feature_n = 0
        #self.Train_iters, self.Validation_data, self.Test_data = self.construct_data( loss_type, batch_size )
        self.Train_data, self.Validation_data, self.Test_data = self.construct_data_o( loss_type, batch_size )
        self.train_data_n
        self.feature_dim

    def map_features(self): # map the feature entries in all files, kept in self.features dictionary
        self.features = {}
        self.users = {}
        self.read_features(self.trainfile)
        self.read_features(self.testfile)
        self.read_features(self.validationfile)
        #print("features_M:", len(self.features))
        return  len(self.features), len(self.users)

    def read_features(self, file): # read a feature file
        # the first one is Y, the second one is user, the rest is features
        f = open( file )
        line = f.readline()
        i = len(self.features)
        j = len(self.users)
        while line:
            items = line.strip().split(' ')
            if items[1] not in self.users:
                self.feature_dim = len(items[2:])
                self.users[items[1]] = j
                j += 1
            for item in items[2:]:
                if item not in self.features:
                    self.features[item] = i
                    i += 1
            line = f.readline()
        f.close()

    def construct_data(self, loss_type, batch_size):
        U_, X_, Y_ , Y_for_logloss= self.read_data(self.trainfile)
        self.feature_n = len(X_[0])
        if loss_type == 'log_loss':
            #Train_data = self.construct_dataset(U_, X_, Y_for_logloss)
            train_dataset = tf.data.Dataset.from_tensor_slices((U_, X_, Y_for_logloss)).batch(batch_size)
        else:
            #Train_data = self.construct_dataset(U_, X_, Y_)
            train_dataset = tf.data.Dataset.from_tensor_slices((U_, X_, Y_)).batch(batch_size)

        train_iters = train_dataset.make_initializable_iterator()

        
        print("# of training:" , len(Y_))
        self.train_data_n = len(Y_)

        U_, X_, Y_ , Y_for_logloss= self.read_data(self.validationfile)
        if loss_type == 'log_loss':
            Validation_data = self.construct_dataset(U_, X_, Y_for_logloss)
        else:
            Validation_data = self.construct_dataset(U_, X_, Y_)
        print("# of validation:", len(Y_))

        U_, X_, Y_ , Y_for_logloss = self.read_data(self.testfile)
        if loss_type == 'log_loss':
            Test_data = self.construct_dataset(U_, X_, Y_for_logloss)
        else:
            Test_data = self.construct_dataset(U_, X_, Y_)
        print("# of test:", len(Y_))


        return train_iters,  Validation_data,  Test_data


    def construct_data_o(self, loss_type, batch_size):
        U_, X_, Y_ , Y_for_logloss= self.read_data(self.trainfile)
        self.feature_n = len(X_[0])
        if loss_type == 'log_loss':
            Train_data = self.construct_dataset(U_, X_, Y_for_logloss)
        else:
            Train_data = self.construct_dataset(U_, X_, Y_)

        print("# of training:" , len(Y_))
        self.train_data_n = len(Y_)

        U_, X_, Y_ , Y_for_logloss= self.read_data(self.validationfile)
        if loss_type == 'log_loss':
            Validation_data = self.construct_dataset(U_, X_, Y_for_logloss)
        else:
            Validation_data = self.construct_dataset(U_, X_, Y_)
        print("# of validation:", len(Y_))

        U_, X_, Y_ , Y_for_logloss = self.read_data(self.testfile)
        if loss_type == 'log_loss':
            Test_data = self.construct_dataset(U_, X_, Y_for_logloss)
        else:
            Test_data = self.construct_dataset(U_, X_, Y_)
        print("# of test:", len(Y_))


        return Train_data,  Validation_data,  Test_data


    def read_data(self, file):
        # read a data file. For a row, the first column goes into Y_;
        # the other columns become a row in X_ and entries are maped to indexs in self.features
        f = open( file )
        U_ = []
        X_ = []
        Y_ = []
        Y_for_logloss = []
        line = f.readline()
        while line:
            items = line.strip().split(' ')
            Y_.append( 1.0*float(items[0]) )

            if float(items[0]) > 0:# > 0 as 1; others as 0
                v = 1.0
            else:
                v = 0.0
            Y_for_logloss.append( v )

            X_.append( [ self.features[item] for item in items[2:]] )
            U_.append([self.users[items[1]]])

            line = f.readline()
        f.close()
        return U_, X_, Y_, Y_for_logloss
    
    def read_data_o(self, file):
        # read a data file. For a row, the first column goes into Y_;
        # the other columns become a row in X_ and entries are maped to indexs in self.features
        f = open( file )
        X_ = []
        Y_ = []
        Y_for_logloss = []
        line = f.readline()
        while line:
            items = line.strip().split(' ')
            Y_.append( 1.0*float(items[0]) )

            if float(items[0]) > 0:# > 0 as 1; others as 0
                v = 1.0
            else:
                v = 0.0
            Y_for_logloss.append( v )

            X_.append( [ self.features[item] for item in items[1:]] )
            line = f.readline()
        f.close()
        return X_, Y_, Y_for_logloss



    def construct_dataset(self, U_, X_, Y_):
        Data_Dic = {}
        X_lens = [ len(line) for line in X_]
        indexs = np.argsort(X_lens)
        Data_Dic['Y'] = [ Y_[i] for i in indexs]
        Data_Dic['X'] = [ X_[i] for i in indexs]
        Data_Dic['U'] = [ U_[i] for i in indexs]
        return Data_Dic
    
    def truncate_features(self):
        """
        #Make sure each feature vector is of the same length
        """
        num_variable = len(self.Train_data['X'][0])
        for i in range(len(self.Train_data['X'])):
            num_variable = min([num_variable, len(self.Train_data['X'][i])])
        # truncate train, validation and test
        for i in range(len(self.Train_data['X'])):
            self.Train_data['X'][i] = self.Train_data['X'][i][0:num_variable]
        for i in range(len(self.Validation_data['X'])):
            self.Validation_data['X'][i] = self.Validation_data['X'][i][0:num_variable]
        for i in range(len(self.Test_data['X'])):
            self.Test_data['X'][i] = self.Test_data['X'][i][0:num_variable]
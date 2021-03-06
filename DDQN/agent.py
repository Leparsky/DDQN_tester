import sys
import numpy as np
import os.path
import keras.backend as K

from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Flatten, Reshape, LSTM, Lambda
from keras.regularizers import l2
from utils.networks import conv_block

class Agent:
    """ Agent Class (Network) for DDQN
    """

    def __init__(self, state_dim, action_dim, lr, tau, dueling, hidden_dim=0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        if hidden_dim == 0:
            self.hidden_dim=self.state_dim
        else:
            self.hidden_dim = hidden_dim
        self.tau = tau
        # Initialize Deep Q-Network
        self.model = self.network(dueling)
        self.model.compile(Adam(lr), 'mse')
        # Build target Q-Network
        self.target_model = self.network(dueling)
        self.target_model.compile(Adam(lr), 'mse')
        self.target_model.set_weights(self.model.get_weights())

    def huber_loss(self, y_true, y_pred):
        return K.mean(K.sqrt(1 + K.square(y_pred - y_true)) - 1, axis=-1)

    def network(self, dueling):
        """ Build Deep Q-Network
        """
        inp = Input(shape=(self.state_dim,))

        # Determine whether we are dealing with an image input (Atari) or not
        '''if(len(self.state_dim) > 2):
            inp = Input((self.state_dim[1:]))
            x = conv_block(inp, 32, (2, 2), 8)
            x = conv_block(x, 64, (2, 2), 4)
            x = conv_block(x, 64, (2, 2), 3)
            x = Flatten()(x)
            x = Dense(256, activation='relu')(x)
        else:'''
        #x = Flatten()(inp) # Слой преобразования данных из 2D представления в плоское
        x = Dense(int(self.hidden_dim), activation='relu')(inp)
        x = Dense(int(self.hidden_dim), activation='relu')(x)


        if(dueling):
            # Have the network estimate the Advantage function as an intermediate layer
            x = Dense(self.action_dim + 1, activation='linear')(x)
            x = Lambda(lambda i: K.expand_dims(i[:,0],-1) + i[:,1:] - K.mean(i[:,1:], keepdims=True), output_shape=(self.action_dim,))(x)
        else:
            x = Dense(self.action_dim, activation='linear')(x)
        return Model(inp, x)

    def transfer_weights(self):
        """ Transfer Weights from Model to Target at rate Tau
        """
        W = self.model.get_weights()
        tgt_W = self.target_model.get_weights()
        for i in range(len(W)):
            tgt_W[i] = self.tau * W[i] + (1 - self.tau) * tgt_W[i]
        self.target_model.set_weights(tgt_W)

    def fit(self, inp, targ):
        """ Perform one epoch of training
        """
        self.model.fit(self.reshape(inp), targ, epochs=1, verbose=0)

    def predict(self, inp):
        """ Q-Value Prediction
        """
        return self.model.predict(self.reshape(inp))

    def target_predict(self, inp):
        """ Q-Value Prediction (using target network)
        """
        return self.target_model.predict(self.reshape(inp))

    def reshape(self, x):
        if len(x.shape) == 1: return np.expand_dims(x, axis=0)
        #else:
        return x

    def saveModel(self, file_path,version):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        if version:
            self.model.save(file_path+version)
        else:
            v=1
            while os.path.exists(file_path+str(v)):
                v += 1
            self.model.save(file_path+str(v))

    def loadModel_versoin(self, file_path, version):
        if version:
            self.model = load_model(file_path + version)
        else:
            v = 1
        while os.path.exists(file_path + str(v)):
            v += 1

    def loadModel(self, file_path):
        self.model = load_model(file_path)
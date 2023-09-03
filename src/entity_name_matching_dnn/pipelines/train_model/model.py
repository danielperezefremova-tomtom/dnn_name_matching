import numpy as np
import tensorflow as tf
from keras.layers import Dropout, concatenate
import os
from keras.activations import relu
import pickle 
import typing

def sparse_encoder(string: str, char_to_index_dict: dict, max_len: int) -> np.array:

    vector = np.zeros(max_len)
    for ind, char in enumerate(string):
        if ind>=max_len:
            break
        vector[ind] = char_to_index_dict[char]
    return vector

def encode_sparsed_list(iterable: iter, char_to_index_dict: dict, max_len: int) -> np.array:
    
    return np.array([sparse_encoder(string, char_to_index_dict, max_len) for string in iterable])

def get_conv_pool(x_input, max_len, suffix, n_grams=[2,3,5,8,13], feature_maps=128):
    branches = []
    for n in n_grams:
        branch = tf.keras.layers.Conv1D(filters=feature_maps, kernel_size=n, activation=relu,
                        name='Conv_' + suffix + '_' + str(n))(x_input)
        branch = tf.keras.layers.MaxPooling1D(pool_size=max_len - n + 1,
                                              strides=1, padding='valid',
                              name='MaxPooling_' + suffix + '_' + str(n))(branch)
        branch = tf.keras.layers.Flatten(name='Flatten_' + suffix + '_' + str(n))(branch)
        branches.append(branch)
    return branches

class char2vecCNN:
    
    def __init__(self,
                input_size: int,
                embedding_dim:int,
                char_to_index:dict):

        self.char_to_index = char_to_index
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.vocabulary_size = len(char_to_index) + 1

        input_sequence = tf.keras.layers.Input(
            shape=self.input_size,
            name='input_sequence')
        
        x = tf.keras.layers.Embedding(
            input_dim=self.vocabulary_size,
            output_dim=self.embedding_dim,
            mask_zero=True
            )(input_sequence)
        
        self.embedding_layer = tf.keras.models.Model(
            inputs=[input_sequence],
            outputs=x
            )

        sequences_embeding = self.embedding_layer(input_sequence)
        
        convolution_branches = get_conv_pool(sequences_embeding, max_len=100, suffix='ngrams')
        convolution_vectors = concatenate(convolution_branches, axis=-1)
        self.convolutions_layer = tf.keras.models.Model(
            inputs=[input_sequence],
            outputs=convolution_vectors
            )

        left_branch_input = tf.keras.layers.Input(shape=(self.input_size, 1), name='left_branch_input')
        right_branch_input = tf.keras.layers.Input(shape=(self.input_size, 1),name='right_branch_input')

        left_branch_features = self.convolutions_layer(left_branch_input)
        right_branch_features = self.convolutions_layer(right_branch_input)

        product_layer = tf.keras.layers.Multiply()(
            [left_branch_features, right_branch_features]
            )
        difference_layer = tf.keras.layers.Subtract()(
            [left_branch_features, right_branch_features]
            )
        concat_layer = tf.keras.layers.Concatenate(axis=1)(
            [left_branch_features, right_branch_features]
            )
        representation = tf.keras.layers.Concatenate(axis=1)(
            [concat_layer, product_layer, difference_layer,]
            )
        
        x = tf.keras.layers.Dense(1024, activation='relu')(representation)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)

        model_output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        self.model = tf.keras.models.Model(
            inputs=[left_branch_input, right_branch_input],
            outputs=model_output
            )
        
        self.model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()]
            )
        
    def fit(self,
        training_pairs,
        target,
        max_epochs,
        patience,
        validation_pairs,
        batch_size,
        callbacks):
        
        #TODO: Find a more elegant way of data preparation (keras loaders?)
        print(training_pairs)
        X1 = self.vectorize(training_pairs[0])
        X2 = self.vectorize(training_pairs[1])

        X1_val = self.vectorize(validation_pairs[0][0])
        X2_val = self.vectorize(validation_pairs[0][1])
        target_val = validation_pairs[1]
        
        _callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)]+ callbacks

        self.model.fit(
            (X1, X2),
            target,
            verbose=1,
            batch_size=batch_size,
            epochs=max_epochs,
            validation_data=((X1_val, X2_val), target_val),
            callbacks=_callbacks)


    def save_model(self, path_to_model):
        '''
        Saves trained model to directory.
    
        :param path_to_model: str, path to save model.
        '''
    
        if not os.path.exists(path_to_model):
            os.makedirs(path_to_model)
        
        self.model.save_weights(path_to_model + '/weights.h5')
    
        with open(path_to_model + '/model.pkl', 'wb') as f:
            pickle.dump([self.embedding_dim, self.char_to_index], f, protocol=2)


    def load_model(self, path):
        '''
        Loads trained model.
    
        :param path: loads model from `path`.
    
        :return c2v_model: Chars2Vec object, trained model.
        '''
        path_to_model = path
    
        with open(path_to_model + '/model.pkl', 'rb') as f:
            structure = pickle.load(f)
            embedding_dim, char_to_index = structure[0], structure[1]
    
        model = char2vecCNN(embedding_dim=embedding_dim, char_to_index=char_to_index)
        model.model.load_weights(path_to_model + '/weights.h5')
        model.model.compile(optimizer='adam', loss='mae')
    
        return model


    def vectorize(self,
                 iterable: typing.List[str]) -> np.array:
        
        X = encode_sparsed_list(
            iterable,
            char_to_index_dict=self.char_to_index,
            max_len=self.input_size
        )
        X = np.expand_dims(X, axis=-1)
        X_pad = tf.keras.preprocessing.sequence.pad_sequences(X)

        return X_pad
    
    #TODO #1: Implement functions to debug the model intermediate layers (embeding, convolutions, representations, etc)
    #TODO #2: Implement functions to evalute the model performance   
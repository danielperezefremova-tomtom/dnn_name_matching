import numpy as np
import tensorflow as tf
from keras.layers import Dropout, concatenate
import os
from keras.activations import relu
import pickle 
import typing

from transformers import AutoTokenizer, TFAutoModel

SEMANTIC_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

def sparse_encoder(
        string: str,
        char_to_index_dict: dict,
        max_len: int) -> np.array:

    vector = np.zeros(max_len)
    try:
        for ind, char in enumerate(string):
            if ind>=max_len:
                break
            vector[ind] = char_to_index_dict[char]

    except TypeError:
        return np.zeros(max_len)
    return vector

def encode_sparsed_list(
        iterable: iter,
        char_to_index_dict: dict,
        max_len: int
        ) -> np.array:
    
    return np.array([
        sparse_encoder(string, char_to_index_dict, max_len) for string in iterable
        ])

def vectorize(
        iterable: typing.List[str],
        char_to_index_dict: dict,
        max_len: int) -> np.array:
        
        X = encode_sparsed_list(
            iterable,
            char_to_index_dict=char_to_index_dict,
            max_len=max_len
        )
        X_expanded = np.expand_dims(X, axis=-1)
        X_pad = tf.keras.preprocessing.sequence.pad_sequences(X_expanded)

        return X_pad

def get_conv_pool(x_input, max_len, suffix, n_grams=[2,3,5,8, 13], feature_maps=128):
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


class TFSentenceTransformer(tf.keras.layers.Layer):
    def __init__(self, model_name_or_path, **kwargs):
        super(TFSentenceTransformer, self).__init__()
        # loads transformers model
        self.model = TFAutoModel.from_pretrained(model_name_or_path, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)    

    def call(self, inputs):
        # runs model on inputs
        model_output = self.model(self.tokenizer.encode(inputs))
        return model_output

class char2vecCNN:
    
    def __init__(self,
                input_size: int,
                embedding_dim:int,
                char_to_index:dict):

        self.char_to_index = char_to_index
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.vocabulary_size = len(char_to_index) + 1
        self.transformer = TFSentenceTransformer(SEMANTIC_MODEL)

        input_sequence = tf.keras.layers.Input(
            shape=self.input_size,
            name='input_sequence')
        
        sequences_embeding = tf.keras.layers.Embedding(
            input_dim=self.vocabulary_size,
            output_dim=self.embedding_dim,
            mask_zero=True
            )(input_sequence)
        
        convolution_branches = get_conv_pool(sequences_embeding, max_len=100, suffix='ngrams')
        concat_convolutions = concatenate(convolution_branches, axis=-1)
        
        self.features_layer = tf.keras.models.Model(
            inputs=[input_sequence],
            outputs=concat_convolutions
            )

        left_branch_input = tf.keras.layers.Input(shape=(self.input_size, 1), name='left_branch_input')
        right_branch_input = tf.keras.layers.Input(shape=(self.input_size, 1),name='right_branch_input')

        left_branch_features = self.features_layer(left_branch_input)
        right_branch_features = self.features_layer(right_branch_input)
        # _, left_branch_semantic_embedding = self.transformer(left_branch_input)
        # _, right_branch_semantic_embedding = self.transformer(right_branch_input)

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
            [concat_layer,
             product_layer,
             difference_layer,
            #  left_branch_semantic_embedding,
            #  right_branch_semantic_embedding
            ]
            )
        
        x = tf.keras.layers.Dense(1024, activation='relu')(representation)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)

        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        self.model = tf.keras.models.Model(
            inputs=[left_branch_input, right_branch_input],
            outputs=output
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
        X1 = vectorize(
            iterable=training_pairs[0],
            char_to_index_dict=self.char_to_index,
            max_len=self.input_size)
        
        X2 = vectorize(
            iterable=training_pairs[1],
            char_to_index_dict=self.char_to_index,
            max_len=self.input_size)

        X1_val = vectorize(
            iterable=validation_pairs[0][0],
            char_to_index_dict=self.char_to_index,
            max_len=self.input_size)
        
        X2_val = vectorize(
            iterable=validation_pairs[0][1],
            char_to_index_dict=self.char_to_index,
            max_len=self.input_size)
        
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
    
    #TODO #1: Implement functions to debug the model intermediate layers (embeding, convolutions, representations, etc)
    #TODO #2: Implement functions to evalute the model performance   
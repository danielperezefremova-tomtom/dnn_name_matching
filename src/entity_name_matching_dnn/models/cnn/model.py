import numpy as np
import tensorflow as tf
import typing
from transformers import AutoTokenizer, TFAutoModel
from ..utils.layers import get_convolutions_pool

SEMANTIC_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

# TODO Review architecture implementation to split by pieces: Encoder, Feature Extrtactor, Classifier, etc

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
                 max_sequence_length: int,
                 embedding_dim: int,
                 max_vocabulary_size: int,
                 char_to_index: dict):

        self.char_to_index = char_to_index
        self.embedding_dim = embedding_dim
        self.max_vocabulary_size = max_vocabulary_size
        self.max_sequence_length = max_sequence_length
        self.transformer = TFSentenceTransformer(SEMANTIC_MODEL)

        input_sequence = tf.keras.layers.Input(
            shape=(1),
            dtype=tf.string,
            name='input_sequence')

        sequences_encoded = tf.keras.layers.TextVectorization(
            # TODO research on most appropiate upper bound for char vocabulary size
            max_tokens=max_vocabulary_size,
            # TODO custom callable could be implemented for str preprocessing
            standardize='lower_and_strip_punctuation',
            split='character',
            ngrams=1,  # TODO n-gram representation >1 may improve classification results vs huge vocabulary size
            output_mode='int',
            output_sequence_length=self.max_sequence_length,
            pad_to_max_tokens=self.max_sequence_length,
            encoding='utf-8',
            vocabulary=list(self.char_to_index.keys())
        )(input_sequence)

        sequences_embeding = tf.keras.layers.Embedding(
            input_dim=self.max_vocabulary_size,
            output_dim=self.embedding_dim,
            mask_zero=True
        )(sequences_encoded)

        convolution_branches = get_convolutions_pool(
            sequences_embeding, max_len=100, suffix='ngrams')
        concat_convolutions = tf.keras.layers.concatenate(convolution_branches, axis=-1)

        self.features_layer = tf.keras.models.Model(
            inputs=[input_sequence],
            outputs=concat_convolutions
        )

        left_branch_input = tf.keras.layers.Input(
            shape=(1),
            dtype=tf.string,
            name='left_branch_input')

        right_branch_input = tf.keras.layers.Input(
            shape=(1),
            dtype=tf.string,
            name='right_branch_input')

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

        X1 = np.asarray(training_pairs[0]).astype(str)
        X2 = np.asarray(training_pairs[1]).astype(str)

        X1_val = np.asarray(validation_pairs[0][0]).astype(str)
        X2_val = np.asarray(validation_pairs[0][1]).astype(str)

        target_val = np.asarray(validation_pairs[1]).astype(int)
        target = np.asarray(target).astype(int)

        _callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience)] + callbacks

        self.model.fit(
            (X1, X2),
            target,
            verbose=1,
            batch_size=batch_size,
            epochs=max_epochs,
            validation_data=((X1_val, X2_val), target_val),
            callbacks=_callbacks)

    # TODO #1: Implement functions to debug the model intermediate layers (embeding, convolutions, representations, etc)
    # TODO #2: Implement functions to evalute the model performance

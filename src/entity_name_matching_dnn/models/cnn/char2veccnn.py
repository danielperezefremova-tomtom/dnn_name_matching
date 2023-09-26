import numpy as np
import tensorflow as tf
import typing
from transformers import AutoTokenizer, TFAutoModel
from ..utils.layers import get_convolutions_pool

# TODO Review architecture implementation to isolate by pieces: Encoder, Feature Extrtactor, Classifier, etc


class TFSentenceTransformer(tf.keras.layers.Layer):
    def __init__(self, model_path):
        super(TFSentenceTransformer, self).__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = TFAutoModel.from_pretrained(model_path, from_pt=True)
        
    def tf_encode(self, inputs):
        def encode(inputs):
            inputs = [x[0].decode("utf-8") for x in inputs.numpy()]
            outputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors='tf')
            return outputs['input_ids'], outputs['token_type_ids'], outputs['attention_mask']
        return tf.py_function(func=encode, inp=[inputs], Tout=[tf.int32, tf.int32, tf.int32])
    
    def process(self, i, t, a):
      def __call(i, t, a):
        model_output = self.model({'input_ids': i.numpy(), 'token_type_ids': t.numpy(), 'attention_mask': a.numpy()})
        return model_output[0]
      return tf.py_function(func=__call, inp=[i, t, a], Tout=[tf.float32])

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = tf.squeeze(tf.stack(model_output), axis=0)
        input_mask_expanded = tf.cast(
            tf.broadcast_to(tf.expand_dims(attention_mask, -1), tf.shape(token_embeddings)),
            tf.float32
        )
        a = tf.math.reduce_sum(token_embeddings * input_mask_expanded, axis=1)
        b = tf.clip_by_value(tf.math.reduce_sum(input_mask_expanded, axis=1), 1e-9, tf.float32.max)
        embeddings = a / b
        embeddings, _ = tf.linalg.normalize(embeddings, 2, axis=1)
        return embeddings
    def call(self, inputs):
        input_ids, token_type_ids, attention_mask = self.tf_encode(inputs)
        model_output = self.process(input_ids, token_type_ids, attention_mask)
        embeddings = self.mean_pooling(model_output, attention_mask)
        return embeddings



class char2vecCNN:

    def __init__(self,
                 max_sequence_length: int,
                 embedding_dim: int,
                 max_vocabulary_size: int,
                 char_to_index: dict,
                 transformer_model_path: str):

        self.char_to_index = char_to_index
        self.embedding_dim = embedding_dim
        self.max_vocabulary_size = max_vocabulary_size
        self.max_sequence_length = max_sequence_length
        self.transformer = TFSentenceTransformer(transformer_model_path)

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
            # TODO n-gram representation >1 may improve classification results vs huge vocabulary size
            ngrams=1,
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

        self.transformer_layer = tf.keras.Model(inputs=input_sequence,
                                                outputs=self.transformer(input_sequence))
        
        # TODO Semantic features not fully integrated. 
        # Issue: Pretrained models from huggingface (keras) does not support abstract
        # tensors to build keras execution graphs so exceptions are raised when concatenating.
        
        left_branch_semantics = self.transformer_layer(left_branch_input)
        right_branch_semantics = self.transformer_layer(right_branch_input)

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
            # left_branch_semantics,
            # right_branch_semantics
             ]
        )

        x = tf.keras.layers.Dense(1024, activation='relu')(representation)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)

        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():

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

        target = np.asarray(target).astype(int)
        target_val = np.asarray(validation_pairs[1]).astype(int)

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

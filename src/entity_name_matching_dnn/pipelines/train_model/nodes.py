import pyspark 
import re
from pyspark.sql.types import ArrayType, StringType, FloatType
from unidecode import unidecode
from pyspark.sql.functions import (col, 
                                   udf, 
                                   array_distinct, 
                                   concat, 
                                   flatten, 
                                   collect_set, 
                                   lit,
                                   rand,
                                   row_number,
                                   explode,
                                   split
                                   )
from functools import reduce
from pyspark.sql.window import Window
from fuzzywuzzy import fuzz
import typing
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from .model import char2vecCNN
from keras.callbacks import History
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix,
                            ConfusionMatrixDisplay)

plt.style.use('ggplot')

def load_data(train_data: pd.DataFrame,
              test_data: pd.DataFrame,
            validation_data: pd.DataFrame,
            vocabulary: dict):
    
    char_to_index = vocabulary

    X1_train = train_data['name_normalized'].values
    X2_train = train_data['alt_name_normalized'].values
    target_train = train_data['target'].values

    X1_val = validation_data['name_normalized'].values
    X2_val = validation_data['alt_name_normalized'].values
    target_val = validation_data['target'].values

    X1_test = test_data['name_normalized'].values
    X2_test = test_data['alt_name_normalized'].values
    target_test = test_data['target'].values

    return [X1_train,
            X2_train,
            target_train,
            X1_val,
            X2_val,
            target_val,
            X1_test,
            X2_test,
            target_test,
            char_to_index]

def train_model(X1_train: np.array,
        X2_train: np.array,
        target_train: np.array,
        X1_val: np.array,
        X2_val: np.array,
        target_val: np.array,
        char_to_index: dict,
        parameters: dict):

    init_parameters = parameters['model_parameters']['init']

    model = char2vecCNN(
        input_size=init_parameters['input_size'],
        embedding_dim = init_parameters['input_size'],
        char_to_index = char_to_index)
    
    fit_parameters = parameters['model_parameters']['fit']
    history = History()

    model.fit(
        training_pairs=(X1_train, X2_train),
        target=target_train,
        max_epochs=fit_parameters['max_epochs'],
        patience=fit_parameters['patience'],
        validation_pairs=((X1_val, X2_val), (target_val)),
        batch_size=fit_parameters['batch_size'],
        callbacks=[history]
    )

    train_loss = history.history['loss']
    val_loss = history.history['loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    fig, ax = plt.subplots(1, 2, figsize=(10,5))

    ax[0].plot(train_loss)
    ax[0].plot(val_loss)
    ax[0].set_title('Loss history')
    ax[0].legend(['Train loss', 'Validation loss'],
                 loc='upper right')

    ax[1].plot(train_acc)
    ax[1].plot(val_acc)
    ax[1].set_title('Accuracy history')
    ax[1].legend(['Train accuracy', 'Validation accuracy'],
                 loc='upper right')
    plt.show()
    plt.close()

    return model, fig

def evaluate_model(
    model, 
    X1_test: np.array,
    X2_test: np.array,
    target_test: np.array,
    parameters: dict):

    prediction_params = parameters['model_parameters']['prediction']
    threshold = prediction_params['threshold']

    X1_test = model.vectorize(X1_test)
    X2_test = model.vectorize(X2_test)

    pred = model.predict((X1_test, X2_test)).flatten()
    pred = pred > threshold

    pred_baseline = np.array([fuzz.token_set_ratio(X1, X2) for X1, X2 in (X1_test, X2_test)])
    pred_baseline = pred_baseline>.75

    true = target_test.flatten()
    true = true > threshold

    cm_model = confusion_matrix(true, pred)
    disp_model = ConfusionMatrixDisplay(confusion_matrix=cm_model)
    plt.close()

    cm_baseline= confusion_matrix(true, pred_baseline)
    disp_baseline = ConfusionMatrixDisplay(confusion_matrix=cm_baseline)
    plt.close()

    classification_log = pd.DataFrame(list(zip(X1_test, X2_test, true, pred, pred_baseline)), 
                                      columns=['name', 'alt_name', 'target', 'model_prediction', 'edit_prediction'])

    return disp_model.figure_, disp_baseline.figure_, classification_log




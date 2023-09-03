from kedro.io import AbstractDataSet
import tensorflow as tf
from typing import Dict, Any
import keras


class TensorflowModelFile(AbstractDataSet):

    def __init__(self, filepath: str, load_args: Dict[str, Any] = None, save_args: Dict[str, Any] = None):
        """
        A dataset to save and load tensorflow models.

        Args:
            data: The tensorflow model to save.

        """

        self.filepath = filepath

        self._load_args = dict(custom_objects=None,
                               compile=True)

        self._save_args = dict(overwrite=True,
                               save_format='keras'
                               )

        if load_args is not None:
            self._load_args.update(load_args)

        if save_args is not None:
            self._save_args.update(save_args)

    def _load(self):
        return tf.keras.models.load_model(
            self.filepath
        )

    def _save(self, data):
        data.model.save(
            self.filepath,
            **self._save_args
        )

    def _exists(self):
        return True if self.data is not None else False

    def _describe(self):
        return None
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import  (load_data,
                     train_model,
                     evaluate_model
                    )



def create_pipeline(**kwargs) -> Pipeline:

    return pipeline(
        [   
            node(
                func=load_data,
                inputs=["train",
                        "test",
                        "validation",
                        "vocabulary_file"],
                outputs=["X1_train",
                        "X2_train",
                        "target_train",
                        "X1_val",
                        "X2_val",
                        "target_val",
                        "X1_test",
                        "X2_test",
                        "target_test",
                        "char_to_index"],
                name="load_training_data",
            ),
            node(
                func=train_model,
                inputs=["X1_train",
                        "X2_train",
                        "target_train",
                        "X1_val",
                        "X2_val",
                        "target_val",
                        "char_to_index",
                        "parameters"],
                outputs=["tensorflow_model", "training_history_loss"],
                name="train_model",
            ),
            node(
                func=evaluate_model,
                inputs=["tensorflow_model",
                        "X1_test",
                        "X2_test",
                        "target_test",
                        "char_to_index",
                        "parameters"],
                outputs=["confussion_matrix_model",
                         "confussion_matrix_baseline",
                         "classification_logs"],
                name="evaluate_model",
            ),

        ]
    )
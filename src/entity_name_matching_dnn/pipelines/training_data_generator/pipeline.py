from kedro.pipeline import Pipeline, node, pipeline
from .nodes import  (
                     normalize_strings,
                     compute_chars_vocabulary,
                     generate_training_examples,
                     load_query_schema,
                     get_schema_name,
                     build_poi_spatial_query,
                     get_response,
                     generate_train_test_split,
                     generate_negative_examples
                    )

from pyspark.sql.window import Window

def create_pipeline(**kwargs) -> Pipeline:

    return pipeline(
        [   
            node(
                func=load_query_schema,
                inputs=["parameters"],
                outputs="query_schema",
                name="load_query_schema",
            ),
            node(
                func=get_schema_name,
                inputs=["df_schema", "query_schema", "parameters"],
                outputs="schemas_name",
                name="get_schema_name",
            ),
            node(
                func=build_poi_spatial_query,
                inputs=[
                    "schemas_name",
                    "parameters",
                ],
                outputs="spatial_query",
                name="build_poi_spatial_query",
            ),
            node(
                func = get_response,
                inputs = ["df_response", "spatial_query", "parameters"],
                outputs = "df_raw_names_sample",
                name = "get_response",
            ),
            node(
                func=normalize_strings,
                inputs=["df_raw_names_sample"],
                outputs="df_normalized",
                name="normalize_pairs",
            ),
            node(
                func=compute_chars_vocabulary,
                inputs=["df_normalized"],
                outputs="vocabulary_file",
                name="compute_vocabulary",
            ),
            node(
                func=generate_training_examples,
                inputs=["df_normalized"],
                outputs=["df_positives", "df_negatives"],
                name="generate_examples",
            ),
            node(
                func=generate_negative_examples,
                inputs=["df_positives", "df_negatives"],
                outputs="df_training",
                name="generate_negative_examples",
            ),
            node(
                func=generate_train_test_split,
                inputs=["df_training", "parameters"],
                outputs=["train", "test", "validation"],
                name="generate_train_test_split",
            ),
            
            
        ]
    )
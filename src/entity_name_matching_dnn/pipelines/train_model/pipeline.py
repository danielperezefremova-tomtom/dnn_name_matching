from kedro.pipeline import Pipeline, node, pipeline
from .nodes import  (
                     
                    )



def create_pipeline(**kwargs) -> Pipeline:

    return pipeline(
        [   
            node(
                func=load_query_schema,
                inputs=["parameters"],
                outputs="query_schema",
                name="load_query_schema",
            )
            
            
        ]
    )
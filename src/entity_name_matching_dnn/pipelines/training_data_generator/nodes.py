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
                                   )
from functools import reduce
from pyspark.sql.window import Window
from fuzzywuzzy import fuzz
import typing
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ...utils.utils import normalize_characters

logger = logging.getLogger("kedro")

def normalize_strings(df: pyspark.sql.DataFrame
              ) -> pyspark.sql.DataFrame:
    
    udf_normalize_chars = udf(normalize_characters)
    df_transformed = df.withColumn('name_normalized', udf_normalize_chars(col('name'))) \
                        .withColumn('alt_name_normalized', udf_normalize_chars(col('alt_name')))
    
    df_filtered = df_transformed.filter((col('name_normalized').isNotNull()) &
                                        (col('name_normalized') != '') & 
                                        (col('alt_name_normalized').isNotNull()) & 
                                        (col('alt_name_normalized') != ''))

    return df_filtered

def get_unique_characters(string: str
                          ) -> list:
    
    unique_characters = list(set(string)) if isinstance(string, str) else []
    return unique_characters

def compute_chars_vocabulary(df: pyspark.sql.DataFrame
                          ) -> dict:

    udf_get_unique_characters = udf(get_unique_characters, ArrayType(StringType()))
    df_transformed = df.withColumn('unique_characters_name', udf_get_unique_characters(col('name_normalized'))) \
                        .withColumn('unique_characters_alt_name', udf_get_unique_characters(col('alt_name_normalized'))) \
                        .withColumn('unique_row_characters', array_distinct(concat(col('unique_characters_name'), col('unique_characters_alt_name'))))
    
    df_aggregated = df_transformed.groupBy().agg(
    array_distinct(flatten(collect_set(col('unique_row_characters')))).alias('char_list')
    )
    
    chracters_list =  df_aggregated.rdd.map(lambda x: x.char_list).collect()[0]
    char_to_index = {char: index+1 for index, char in enumerate(chracters_list)} # 0 index is reserved for padding

    return char_to_index

def generate_training_examples(df: pyspark.sql.DataFrame
                               ) -> pyspark.sql.DataFrame:
    
    #TODO: Implement more sophisticated negative examples

    positive_examples = df.toPandas()
    positive_examples['target'] = 1

    negative_examples = positive_examples.copy()
    negative_examples['target'] = 0

    np.random.seed(0)
    indexes = positive_examples.index.tolist()
    np.random.shuffle(indexes)

    negative_examples['alt_name'] = positive_examples.loc[indexes, 'alt_name'].values
    negative_examples['alt_name_normalized'] = positive_examples.loc[indexes, 'alt_name_normalized'].values

    training_data = pd.concat([positive_examples, negative_examples], axis=0)

    return training_data

def load_query_schema(parameters: typing.Dict) -> str:
    """
        Read '[orbis|genesis]_schema_query' parameter in queries.yml. Then format it with
        country defined in superfluous_pois.yml
        Parameters
            parameters (Dict): Parameters dictionary
        Returns:
            query (str): query formatted with the country
    """
    provider = parameters["provider"]
    schema_query = parameters["query_parameters"][provider]["schema_query"]
    query = parameters[schema_query].format(country=parameters["country"].lower())
    logger.info(query)
    return query

def get_schema_name(df_dict, query_schema: str, parameters: typing.Dict) -> str:
    """
        Extract country schema in the input dataframe for a given country
        Parameters
            df_dict (Dict[str, DataFrame]): spark dataframe containing the result of
                executing the query '[orbis|genesis]_schema_query' against the server
            query_schema (str): dummy string to ensure that this node is executed after the previous one.
            parameters (Dict): parameters dictionary
        Returns:
            schema (str): schema name in the database for the country defined in parameters. It can be a
            comma separated string if the country has more than one schema (usa and canada for genesis)
    """
    country = parameters["country"]
    rows = df_dict['query'].collect()

    if rows:
        schemas = list(map(lambda row: row["schema_name"], rows))
        schema = ','.join(schemas)
        logger.info(f"Available schemas: {schemas}")
        logger.info(f"Selected schema (comma separated): {schema}")
        return schema
    else:
        raise RuntimeError(f"Schema not found for country '{country}'")

def build_poi_spatial_query(
    schemas: str,
    parameters: dict,
) -> str:
    
    country = parameters['country']
    provider = parameters["provider"]
    pois_number = parameters["pois_number"]

    
    data_query = parameters["query_parameters"][provider]["data_query"]
    query_template = parameters[data_query]

    spatial_query = query_template.format(
        country = country,
        schema = schemas,
        pois_number = pois_number
    )

    return spatial_query

def unionAll(*dfs):
    return reduce(pyspark.sql.DataFrame.unionAll, dfs)

def get_response(df_response: typing.Dict[str, pyspark.sql.DataFrame],
                                dummy_input,
                                parameters:dict) -> pyspark.sql.DataFrame:

    run_id = parameters['run_id']
    df_response = unionAll(
            *[df for df in df_response.values()]
            )
    
    df_response = df_response.withColumn('run_id', lit(run_id))
    
    return df_response


def generate_train_test_split(df: pd.DataFrame,
                                 parameters:dict
                                 ) -> pd.DataFrame:
    
    """Produces a 60%, 20%, 20% split for training, validation and test sets.

    Args:
        df (pd.DataFrame): 

    Returns:
        np.array: pd.DataFrame
    """
    np.random.seed(0)
    train, validation, test = np.split(df.sample(frac=1),
                                 [int(.6*len(df)), int(.8*len(df))])

    logger.info(f'Generated: {train.shape[0]} train pairs | {test.shape[0]} test pairs | {validation.shape[0]} val pairs')

    return train, test, validation
    






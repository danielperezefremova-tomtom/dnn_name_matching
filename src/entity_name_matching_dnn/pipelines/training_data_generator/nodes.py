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

logger = logging.getLogger("kedro")


def normalize_characters(text: str) -> str:
    
    if text is None:
        return ''

    strange_char_space = r"!#$%&()*+,./:;<=>?@[\]^_{|}~ºª®-"
    char_2_remove = "\"'^`"
    space_mask = "                                "
    punctuation_table = str.maketrans(strange_char_space, space_mask,
                                        char_2_remove)
    text = text.lower()
    text = text.strip()
    # text = unidecode(text)
    text = text.translate(punctuation_table)
    text = re.sub(r' +', ' ', text)
    text = text.strip()
    text = text.lower()

    return text

def normalize_strings(df: pyspark.sql.DataFrame
              ) -> pyspark.sql.DataFrame:
    
    udf_normalize_chars = udf(normalize_characters)
    df_transformed = df.withColumn('name_normalized', udf_normalize_chars(col('name'))) \
                        .withColumn('alt_name_normalized', udf_normalize_chars(col('alt_name')))

    return df_transformed

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

def get_similar_neg_examples(df: pyspark.sql.DataFrame
                               ) -> pyspark.sql.DataFrame:

    udf_token_set_ratio = udf(fuzz.token_set_ratio)
    w = Window().partitionBy(lit('country')).orderBy(lit('country'))

    df_indexed = df.withColumn("index", row_number().over(w))
    df_name = df_indexed.select('name_normalized', 'index')
    df_alt_name = df_indexed.select('alt_name_normalized').orderBy(rand()).withColumn("index", row_number().over(w))

    df_neg_examples = df_name.join(df_alt_name, on='index')
    df_neg_examples = df_neg_examples.withColumn('similarity_score', udf_token_set_ratio(col('name_normalized'), col('alt_name_normalized')))
    df_neg_examples = df_neg_examples.filter((col('similarity_score')>50) & (col('similarity_score')<100))

    return df_neg_examples

def generate_neg_examples(df: pyspark.sql.DataFrame
                               ) -> pyspark.sql.DataFrame:
    
    df_neg_examples = reduce(unionAll, [get_similar_neg_examples(df) 
                              for _ in range(100)])
    
    return df_neg_examples

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


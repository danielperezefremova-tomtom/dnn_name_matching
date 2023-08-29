from typing import Any, Dict
from kedro.io import AbstractDataSet, DataSetError
from pyspark.sql import DataFrame, SparkSession


class SparkJDBCQueryDataSet(AbstractDataSet[DataFrame, DataFrame]):

    def __init__(self, credentials: Dict, filepath: str = None):
        self.credentials = credentials
        self.filepath = filepath
        self.query = None
        self.load_query_from_path()

    def load_query_from_path(self):
        with open(self.filepath, "r") as file:
            self.query = file.read()

    def _load(self) -> DataFrame:
        host = self.credentials["host"]
        port = self.credentials["port"]
        database = self.credentials["database"]
        user = self.credentials["user"]
        password = self.credentials["password"]

        spark = self._get_spark()
        df = spark.read \
            .format("jdbc") \
            .option("url", f"jdbc:postgresql://{host}:{port}/{database}") \
            .option("driver", "org.postgresql.Driver") \
            .option("query", self.query) \
            .option("user", user) \
            .option("password", password) \
            .load()
        return df

    def _save(self, df: DataFrame) -> None:
        raise DataSetError("Method 'save' not supported in SparkJDBCQueryDataSet")

    def _describe(self) -> Dict[str, Any]:
        return dict(
            host=self.credentials["host"],
            port=self.credentials["port"],
            database=self.credentials["database"],
            user=self.credentials["user"],
            password="*****",
            filepath=self.filepath,
            query=self.query
        )

    @staticmethod
    def _get_spark():
        return SparkSession.builder.getOrCreate()
from kedro.framework.hooks import hook_impl
from pyspark import SparkConf
from pyspark.sql import SparkSession
import logging
import json 
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
import os
from kedro.config import MissingConfigException
from typing import Any, Dict
from kedro.io import DataCatalog, MemoryDataSet

DEFAULT_VAULT = "KedroSecrets"
log = logging.getLogger("kedro")

class SparkHooks:
    @hook_impl
    def after_context_created(self, context) -> None:
        """Initialises a SparkSession using the config
        defined in project's conf folder.
        """

        # Load the spark configuration in spark.yaml using the config loader
        parameters = context.config_loader.get("spark*", "spark*/**")
        spark_conf = SparkConf().setAll(parameters.items())

        # Initialise the spark session
        spark_session_conf = (
            SparkSession.builder.appName(context.project_path.name)
            .enableHiveSupport()
            .config(conf=spark_conf)
        )
        _spark_session = spark_session_conf.getOrCreate()
        _spark_session.sparkContext.setLogLevel("WARN")

class AzureSecretsHook:
    @hook_impl
    def after_context_created(self, context) -> None:

        key_vault_name = os.environ.get("KEY_VAULT_NAME", DEFAULT_VAULT)

        try:
            client = SecretClient(
                vault_url=f"https://{key_vault_name.lower()}.vault.azure.net",
                credential=DefaultAzureCredential()
            )
            secrets = client.list_properties_of_secrets()

            azure_creds = {
                secret.name: client.get_secret(secret.name).value
                for secret in secrets
                if secret.enabled
            }
        except Exception as e:
            log.warning("azure keyvault secrets not loaded")
            log.warning(str(e))
            azure_creds = {}

        try:
            credentials_in_conf = context.config_loader["credentials"]
        except MissingConfigException:
            credentials_in_conf = {}

        context.config_loader["credentials"] = {
            **credentials_in_conf,
            **azure_creds,
        }

    
        # Generic credentials
        country = context.config_loader["parameters"].get("country").lower()
        provider = context.config_loader["parameters"].get("provider")
        has_custom_credentials = context.config_loader["parameters"].get("has_custom_credentials")
        custom_credentials = context.config_loader["parameters"].get("custom_credentials")

        keyvault_key = context.config_loader["parameters"].get(f"{country}_{provider}")
        credentials = azure_creds.get(keyvault_key, "{}")
        credentials_dict = json.loads(credentials)

        if has_custom_credentials:
            logging.info('Applying custom credentials for Ad hoc measurement')
            for key, value in custom_credentials.items():
                if value:
                    credentials_dict[key] = value
                    logging.info(f'{key} value overwritten')
                else: 
                    continue
                
        context.config_loader["credentials"]["generic"] = credentials_dict

class ParamsToBoolHook:
    """
    Extends CLI params parsing functionality to allow bool parsing.

    Example:

    .. code-block:: bash

       kedro run --params has_custom_credentials:True
    """

    def __init__(self):
        self.dict_extra_params: Dict[str, Any] = {}

    @hook_impl
    def before_pipeline_run(
        self, run_params: Dict[str, Any], catalog: DataCatalog
    ) -> None:
        for param, param_value in run_params.items():
            try:
                transformed_param_value = (
                    True
                    if param_value.lower() == "true"
                    else False
                    if param_value.lower() == "false"
                    else param_value
                )
            except AttributeError:
                transformed_param_value = param_value

            self.dict_extra_params[param] = transformed_param_value

        if run_params == self.dict_extra_params:
            return None

        parameters = catalog.load("parameters")

        for param, param_value in self.dict_extra_params.items():
            catalog.add_feed_dict(
                {f"params:{param}": MemoryDataSet(param_value)}, replace=True
            )
            parameters[param] = param_value

        catalog.add_feed_dict({"parameters": parameters}, replace=True)
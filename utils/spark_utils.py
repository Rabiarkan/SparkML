import json
from pyspark.sql import SparkSession


def get_spark_session():

    config_path = "/home/ubuntu/spark-cases/SparkML/utils/spark_config.json"


    with open(config_path, "r") as f:
        config = json.load(f)
        
    builder = SparkSession.builder.appName(config.get("appName", "RA-SPARK")).master(config.get("master", "local[*]"))

    for key, value in config.get("configs", {}).items():
        builder = builder.config(key, value)

    spark = builder.getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    
    return spark 



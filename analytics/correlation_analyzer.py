from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame
from typing import List
import pandas as pd

from visualization.spark_visualize import SparkVisualization


class SparkCorrelationAnalyzer:

    @staticmethod
    def analyze_correlation(dataframe: DataFrame, feature_cols: List[str], method: str = "pearson") -> pd.DataFrame:
        """
        Performs correlation analysis using PySpark.

        Parameters:
          dataframe (DataFrame): The input dataframe.
          feature_cols (List): List of column names for variables.
          method (str): Correlation method, e.g., 'pearson' or 'spearman'. Default is 'pearson'.

        Returns:
          Plot: Correlation Matrix Plot.
        """

        # Combine variables into a vector
        vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="vec_features")
        df_assembled = vector_assembler.transform(dataframe)

        # Calculate the correlation matrix
        corr_matrix = Correlation.corr(df_assembled,"vec_features", method)

        # Convert dataframe to pandas df
        matrix_array = corr_matrix.select(f"{method}(vec_features)").head()[0].toArray()

        correlation_matrix_pd = pd.DataFrame(matrix_array, columns=feature_cols, index=feature_cols)

        SparkVisualization.visualize_correlation_matrix(correlation_matrix_pd, feature_cols)



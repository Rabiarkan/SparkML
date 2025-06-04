from pyspark.sql import functions as F
from pyspark.sql import types as T

class SparkStatistics:

    @staticmethod
    def data_describe(spark_session, dataframe, columns=None):
        """
            Calculates and displays descriptive statistics of the given PySpark DataFrame.

            Parameters:
                spark_session (pyspark.sql.SparkSession): SparkSession object
                dataframe (DataFrame): The input PySpark dataframe.
                columns (List): The column to describe.
            Returns:
                Displays result.
        """
        if columns is None:
            describe_df = dataframe.describe()
        else:
            describe_df = dataframe.select(columns).describe()

        metrics = describe_df.columns[1:]  # except summary col
        summary_col = describe_df.select("summary").rdd.flatMap(lambda x: x).collect()

        transposed_data = []
        for col_name in metrics:
            values = describe_df.select(col_name).rdd.flatMap(lambda x: x).collect()
            row = [col_name] + values
            transposed_data.append(row)

        transposed_df = spark_session.createDataFrame(transposed_data, ["Field"] + summary_col)
        transposed_df.show(100, False)


    @staticmethod
    def groupBy_counts(dataframe, column=None):
        """
            Performs a group by operation on each column of the PySpark DataFrame,
            counts the occurrences, sorts the results in descending order, and displays the results.

            Parameters:
                dataframe (DataFrame): The input PySpark dataframe.
                column (str): The column to group by.
            Returns:
                Displays result.
        """
        if column is None:
            for col in dataframe.columns:
                dataframe.groupBy(col).count().sort(F.desc("count")).show(50, False)
        else:
            dataframe.groupBy(column).count().sort(F.desc("count")).show(50, False)


    @staticmethod
    def statistics(dataframe, column, stat_method):
        """
            Calculates and displays a specified statistical measure for a given column in the PySpark DataFrame.

            Parameters:
                dataframe (DataFrame): The input PySpark dataframe.
                column (str): The name of the column for which to calculate the statistical measure.
                stat_method (str): The statistical measure to calculate. Options are: 'mode', 'mean', 'median', 'kurtosis', 'skewness', 'variance' or 'stddev'.
            Returns:
                Displays result.
            Raises:
                ValueError: If an invalid stat_method name is provided.
        """
        stat_method = stat_method.lower()

        if stat_method == "mode":
            result = dataframe.select(F.mode(F.col(column)).alias("Mode(" + column + ")"))
        elif stat_method == "mean":
            result = dataframe.select(F.mean(F.col(column)).alias("Mean(" + column + ")"))
        elif stat_method == "median":
            result = dataframe.select(F.median(F.col(column)).alias("Median(" + column + ")"))
        elif stat_method == "kurtosis":
            result = dataframe.select(F.kurtosis(F.col(column)).alias("Kurtosis(" + column + ")"))
        elif stat_method == "skewness":
            result = dataframe.select(F.skewness(F.col(column)).alias("Skewness(" + column + ")"))
        elif stat_method == "variance":
            result = dataframe.select(F.variance(F.col(column)).alias("Variance(" + column + ")"))
        elif stat_method == "stddev":
            result = dataframe.select(F.stddev(F.col(column)).alias("StdDev(" + column + ")"))
        else:
            raise ValueError(
                "Invalid method name. Please choose one of: mode, mean, median, kurtosis, skewness, variance or stddev.")

        return result


    @staticmethod
    def statistics_all_columns(spark_session, dataframe):
        """
            Calculates and displays various statistical measures for all numeric columns in the given PySpark DataFrame.

            Parameters:
             spark_session (SparkSession): The Spark session used to create the resulting DataFrame.
             dataframe (DataFrame): The input PySpark DataFrame containing the data to analyze.

            Returns:
             DataFrame: A new DataFrame where each row contains the column name and the computed statistics
                        (mode, mean, median, stddev, variance, kurtosis, skewness) for that column.
         """

        numeric_columns = [col_name for col_name, col_type in dataframe.dtypes if
                           isinstance(dataframe.schema[col_name].dataType, T.NumericType)]

        result_rows = []

        for col in numeric_columns:
            mode = SparkStatistics.statistics(dataframe, col, "mode").first()[0]
            mean = SparkStatistics.statistics(dataframe, col, "mean").first()[0]
            median = SparkStatistics.statistics(dataframe, col, "median").first()[0]
            stddev = SparkStatistics.statistics(dataframe, col, "stddev").first()[0]
            variance = SparkStatistics.statistics(dataframe, col, "variance").first()[0]
            kurtosis = SparkStatistics.statistics(dataframe, col, "kurtosis").first()[0]
            skewness = SparkStatistics.statistics(dataframe, col, "skewness").first()[0]

            result_rows.append((col, mode, mean, median, stddev, variance, kurtosis, skewness))

        result_df = spark_session.createDataFrame(result_rows,
                                          ["Column", "mode", "mean", "median", "stddev", "variance", "kurtosis",
                                           "skewness"])

        return result_df



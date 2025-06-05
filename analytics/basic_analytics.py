from pyspark.sql import functions as F
from pyspark.sql import types as T


class SparkAnalytics:

    @staticmethod
    def ydataprofiling(dataframe, report_name):

        from ydata_profiling import ProfileReport

        report = ProfileReport(dataframe)

        file_name = f"{report_name}_spark_profile.html"

        report.to_file(file_name)


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
                dataframe.groupBy(col).count().sort(F.desc("count")).show(25, False)
        else:
            dataframe.groupBy(column).count().sort(F.desc("count")).show(25, False)

    
    @staticmethod
    def groupBy_counts_and_freq(dataframe, column):
        
        grouped_df = (dataframe.groupBy(column)
            .agg(F.count("*").alias("Count"))  
        )

        total_count = dataframe.count()

        grouped_df = grouped_df.withColumn(
            "Frequency (%)", F.round((F.col("Count") / total_count) * 100, 1)
        )
        
        grouped_df = grouped_df.orderBy(F.col("Count").desc())

        grouped_df.show(25, False)
        

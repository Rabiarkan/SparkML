import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

import pandas as pd
import numpy as np

class SparkVisualization:

    @staticmethod
    def visualize_correlation_matrix(correlation_matrix, feature_cols):
        """
            Visualizes the correlation matrix using seaborn.

            Parameters:
                correlation_matrix: Pandas DataFrame containing the correlation matrix.
            Returns:
                Plot: Correlation matrix Plot.
        """

        num = len(feature_cols)
        plt.figure(figsize=(num, num))

        mask = np.triu(np.ones_like(correlation_matrix, dtype=np.bool))

        heatmap = sns.heatmap(correlation_matrix, mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')

        heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=16)


        #sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        #plt.title("Correlation Matrix")
        plt.show()


    @staticmethod
    def visualize_outlier_boxplot(dataframe, outlier_cleaned_dataframe, column):
        """
            Visualizes the box plots of a specific column in both the original dataframe and the outlier-cleaned dataframe.

            Parameters:
                dataframe (DataFrame): The original dataframe containing the data.
                outlier_cleaned_dataframe (DataFrame): The dataframe with outliers removed.
                column (str): The name of the column to visualize.
            Returns:
                Plot: Outlier Boxplot.
        """

        pd_dataframe = dataframe.select(column).toPandas()
        pd_outlier_cleaned_dataframe = outlier_cleaned_dataframe.select(column).toPandas()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        sns.boxplot(data=pd_dataframe, ax=axs[0]).set(title="Box Plot(" + column + ")")
        sns.boxplot(data=pd_outlier_cleaned_dataframe, ax=axs[1]).set(title="Clean Box Plot(" + column + ")")

        plt.show()


    @staticmethod
    def visualize_feature_importance_scores(rfc_model, features_list):
        """
            Visualizes the feature importance scores using a bar plot.

            Parameters:
                rfc_model (object): A trained Random Forest Classifier model.
                features_list (list): A list of feature names corresponding to the importance scores.
            Returns:
                Plot: Feature importance Score plot.
        """

        # Create a list of tuples containing feature names and their importances
        importance_data = list(zip(features_list, rfc_model.featureImportances))
        # Convert the list of tuples into a DataFrame
        features_df = pd.DataFrame(importance_data, columns=['name', 'Importance'])

        # Sort the DataFrame by Importance in ascending order
        features_df.sort_values("Importance", ascending=True, inplace=True)

        # Create a bar plot using seaborn
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=features_df, x='Importance', y='name', hue='name', palette='viridis', dodge=False, legend=False)
        plt.title('Feature Importance Plot')
        plt.xlabel("Importance Score")
        plt.ylabel("Variable Importance")

        # Annotate each bar with its importance value
        for p in ax.patches:
            ax.annotate(f'{p.get_width():.4f}',
                        (p.get_width(), p.get_y() + p.get_height() / 2.),
                        ha='center', va='center',
                        fontsize=10, color='black',
                        xytext=(10, 0),
                        textcoords='offset points')

        plt.show()


    @staticmethod
    def visualize_linechart(dataframe, columns=None, x_axis=None):
        """
            Visualizes line plots of columns.

            Parameters:
                dataframe (DataFrame): The original dataframe containing the data.
                columns: List of column names to plot, if None plots all columns
                x_axis: Column name to use for the x-axis, if None uses the DataFrame index.
            Returns:
                Plot: Line chart.
        """

        pd_dataframe = dataframe.toPandas()

        fig = go.Figure()

        if columns is None:
            columns = pd_dataframe.columns
        elif isinstance(columns, str):
            columns = [columns]

        for col in columns:
            fig.add_trace(go.Scatter(
                x=pd_dataframe[x_axis] if x_axis else pd_dataframe.index,
                y=pd_dataframe[col],
                mode='lines',
                name=col
            ))

        fig.show()
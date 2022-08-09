import plotly.graph_objects as go
from sos_trades_core.tools.post_processing.post_processing_tools import align_two_y_axes, format_currency_legend
from sos_trades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import \
    InstantiatedPlotlyNativeChart
import pandas as pd
import numpy as np


class CommonCharts(InstantiatedPlotlyNativeChart):

    """ Class to host standard post post processing charts templates
    """

    def __init__(self):
        super().__init__(go.Figure())
        self.default_chart = InstantiatedPlotlyNativeChart(go.Figure())
        self.default_legend = self.default_chart.get_default_legend_layout()

    def generate_bar_chart(self, data_df:pd.DataFrame, x_axis_column:str, y_axis_column_list:list, x_axis_title:str='', y_axis_title:str='', chart_name:str='', ticksuffix:str='', annotation_upper_left:dict={}, annotation_upper_right:dict={}, labels_dict:dict={}, annotations:list=[], updatemenus:list=[], barmode:str='stack', text_inside_bar:bool=False, add_cumulated:bool=False, column_val_cum_sum:str=None, showlegend:bool=True)->InstantiatedPlotlyNativeChart:
        """Generate a bar chart from data in a dataframe

        Args:
            data_df (pd.DataFrame): dataframe containing data. 
            x_axis_column (str): dataframe column name for the x-axis
            y_axis_column_list (list): dataframe columns name for the y-axis. Each column will result in a separate serie
            x_axis_title (str, optional): Title for x-axis. Defaults to ''.
            y_axis_title (str, optional): Title for y-axis. Defaults to ''.
            chart_name (str, optional): Chart name. Defaults to ''.
            ticksuffix (str, optional): Ticksuffix to display units after the values for exemple. Defaults to ''.
            annotation_upper_left (dict, optional): annotation to put in the upper left corner of the chart. Defaults to {}.
            annotation_upper_right (dict, optional): annotation to put in the upper right corner of the chart.. Defaults to {}.
            labels_dict (dict, optional): _description_. Defaults to {}.
            annotations (list, optional): _description_. Defaults to [].
            updatemenus (list, optional): _description_. Defaults to [].
            barmode (str, optional): stack to have stacked bar, group to have a separate bar for each serie. Defaults to 'stack'.
            text_inside_bar (bool, optional): _description_. Defaults to False.
            add_cumulated (bool, optional): True to add a cumulated serie for te value. Defaults to False.
            column_val_cum_sum (str, optional): Column to use as the cumulated value. Defaults to None.
            showlegend (bool, optional): Possibility to show or hide the legend. Defaults to True.

        Returns:
            InstantiatedPlotlyNativeChart: Plotly Instanciated chart with data
        """        
        # Create figure
        fig = go.Figure()

        for col in y_axis_column_list:
            if col in data_df:
                fig.add_trace(
                    go.Bar(
                        x=data_df[x_axis_column].values.tolist(),
                        y=data_df[col].values.tolist(),
                        name=f'{labels_dict.get(col,col)}' if len(
                            labels_dict) != 0 else col,
                        text=col if text_inside_bar else '',
                        textposition='inside',
                        xaxis='x',
                        yaxis='y',
                        visible=True,
                    )
                )

        if add_cumulated:
            fig.add_trace(
                go.Scatter(
                    x=data_df[x_axis_column].values.tolist(),
                    y=data_df[column_val_cum_sum].cumsum().values.tolist(),
                    name=f'Cumulative {labels_dict.get(column_val_cum_sum,column_val_cum_sum)}' if len(
                        labels_dict) != 0 else f'Cumulative {column_val_cum_sum}',
                    xaxis='x',
                    yaxis='y2',
                    visible=True,
                    mode='lines',
                )
            )

        fig.update_layout(
            xaxis=dict(
                title=x_axis_title,
                titlefont_size=12,
                tickfont_size=10,
                automargin=True
            ),
            yaxis=dict(
                title=y_axis_title,
                titlefont_size=12,
                tickfont_size=10,
                ticksuffix=f'{ticksuffix}',
                automargin=True,
            ),
            # legend=self.default_legend,
            barmode=barmode,
            autosize=True,
            # legend=dict(yanchor="bottom", y=-0.1, orientation='h'),
        )

        if len(y_axis_column_list) > 1:
            fig.update_layout(
                showlegend=showlegend,
            )

        if len(annotations) != 0:
            fig.update_layout(annotations=annotations[0])

        if len(updatemenus) != 0:
            fig.update_layout(updatemenus=updatemenus)

        if add_cumulated:
            fig.update_layout(
                yaxis2=dict(
                    title=f'Cumulative {column_val_cum_sum}',
                    titlefont_size=12,
                    tickfont_size=10,
                    ticksuffix=f'{ticksuffix}',
                    automargin=True,
                    anchor="x",
                    overlaying="y",
                    side="right",
                ),
            )

        new_chart = None
        if len(fig.data):
            if add_cumulated:
                fig = align_two_y_axes(fig, GRIDLINES=4)

            # Create native plotly chart
            new_chart = InstantiatedPlotlyNativeChart(
                fig=fig, chart_name=chart_name, default_legend=False)
            new_chart.annotation_upper_left = annotation_upper_left
            new_chart.annotation_upper_right = annotation_upper_right
            new_chart.set_csv_data_from_dataframe(
                data_df)
        return new_chart

    def generate_lines_chart(self, data_df:pd.DataFrame, x_axis_column:str, y_axis_column_list:list, x_axis_title:str='', y_axis_title:str='', chart_name:str='', ticksuffix:str='', annotation_upper_left:dict={}, annotation_upper_right:dict={}, labels_dict:dict={}, mode:str='lines', textposition:str="top center")->InstantiatedPlotlyNativeChart:
        """Generate a line chart from data in a dataframe

        Args:
            data_df (pd.DataFrame): dataframe containing data. 
            x_axis_column (str): dataframe column name for the x-axis
            y_axis_column_list (list): dataframe columns name for the y-axis. Each column will result in a separate serie
            x_axis_title (str, optional): Title for x-axis. Defaults to ''.
            y_axis_title (str, optional): Title for y-axis. Defaults to ''.
            chart_name (str, optional): Chart name. Defaults to ''.
            ticksuffix (str, optional): Ticksuffix to display units after the values for exemple. Defaults to ''.
            annotation_upper_left (dict, optional): annotation to put in the upper left corner of the chart. Defaults to {}.
            annotation_upper_right (dict, optional): annotation to put in the upper right corner of the chart.. Defaults to {}.
            labels_dict (dict, optional): _description_. Defaults to {}.
            mode (str, optional): _description_. Defaults to 'lines'.
            textposition (str, optional): _description_. Defaults to "top center".

        Returns:
            InstantiatedPlotlyNativeChart: Plotly Instanciated chart with data
        """        
        # Create figure
        fig = go.Figure()

        for column in y_axis_column_list:
            if column in data_df:
                fig.add_trace(
                    go.Scatter(
                        x=data_df[x_axis_column].values.tolist(),
                        y=data_df[column].values.tolist(),
                        name=f'{labels_dict.get(column,column)}' if len(
                            labels_dict) != 0 else column,
                        xaxis='x',
                        yaxis='y',
                        visible=True,
                        mode=mode,
                        text=data_df[column].values.tolist(),
                        textposition=textposition
                    )
                )

        fig.update_layout(
            autosize=True,
            xaxis=dict(
                title=x_axis_title,
                titlefont_size=12,
                tickfont_size=10,
                automargin=True
            ),
            yaxis=dict(
                title=y_axis_title,
                titlefont_size=12,
                tickfont_size=10,
                ticksuffix=f'{ticksuffix}',
                automargin=True,

            ),
        )

        new_chart = None
        if len(fig.data):
            # Create native plotly chart
            new_chart = InstantiatedPlotlyNativeChart(
                fig=fig, chart_name=chart_name, default_legend=False)
            new_chart.annotation_upper_left = annotation_upper_left
            new_chart.annotation_upper_right = annotation_upper_right
            new_chart.set_csv_data_from_dataframe(
                data_df)
        return new_chart

    def generate_table_chart(self, data_df, chart_name):
        '''
        data_df : dataframe with data to plot
        '''
        # Create figure
        fig = go.Figure()

        fig.add_trace(
            go.Table(
                header=dict(
                    values=data_df.columns.tolist(),
                    fill_color='midnightblue',
                    align='center',
                    font_color='white',
                ),
                cells=dict(
                    values=data_df.T.values.tolist(),
                    fill_color='floralwhite',
                    align='center',
                ),
            )
        )

        fig.update_layout(
            title_text=chart_name,
            showlegend=False,
            autosize=True,
            height=data_df.shape[0] * 30 + 250,
        )

        new_chart = None
        if len(fig.data):
            # Create native plotly chart
            new_chart = InstantiatedPlotlyNativeChart(
                fig=fig, chart_name=chart_name, default_legend=False)
        return new_chart

    def generate_sunburst_chart(self, sunburst_labels, sunburst_parents, sunburst_values, sunburst_text, chart_name='', branchvalues="total", textinfo='label+text', hoverinfo='label+text+percent parent', ):

        # Create figure
        fig = go.Figure()

        # category bar
        fig.add_trace(
            go.Sunburst(
                # ids=rc_percent_BOM['RC_components'].append(rc_percent_BOM['components']),
                labels=sunburst_labels,
                parents=sunburst_parents,
                values=sunburst_values,
                text=sunburst_text,
                hovertext=sunburst_text,
                branchvalues=branchvalues,
                textinfo=textinfo,
                hoverinfo=hoverinfo
                # domain=dict(column=0)
            )
        )

        # Chart Layout update
        fig.update_layout(
            autosize=True,
            margin=dict(t=60, l=0, r=0, b=0),
        )

        # Create native plotly chart
        chart_name = chart_name
        new_chart = InstantiatedPlotlyNativeChart(
            fig=fig, chart_name=chart_name)

        return new_chart
    def generate_lines_chart_by_category(self, data_df, column_with_categories, x_axis_column, y_axis_column_list, x_axis_title='', y_axis_title='', chart_name='', ticksuffix='', annotation_upper_left={}, annotation_upper_right={}, labels_dict={}, mode='lines', textposition="top center"):
        '''
        data_df : dataframe with data to plot but these data are repeated as many times as number of categories
        x_axis_column : string column name of x axis
        y_axis_column_list : list columns names for y bar to plot  
        '''
        
        categories_list=data_df[column_with_categories].unique()
        # Create figure
        fig = go.Figure()
        for category in categories_list:
            cf_df_by_cat=data_df[data_df[column_with_categories]==category]
            for column in y_axis_column_list:
                if column in data_df:
                    fig.add_trace(
                        go.Scatter(
                            x=cf_df_by_cat[x_axis_column].values.tolist(),
                            y=cf_df_by_cat[column].values.tolist(),
                            name=f'{category}',
                            xaxis='x',
                            yaxis='y',
                            visible=True,
                            mode=mode,
                            text=cf_df_by_cat[column].values.tolist(),
                            textposition=textposition
                        )
                    )

        fig.update_layout(
            autosize=True,
            xaxis=dict(
                title=x_axis_title,
                titlefont_size=12,
                tickfont_size=10,
                automargin=True
            ),
            yaxis=dict(
                title=y_axis_title,
                titlefont_size=12,
                tickfont_size=10,
                ticksuffix=f'{ticksuffix}',
                automargin=True,

            ),
        )

        new_chart = None
        if len(fig.data):
            # Create native plotly chart
            new_chart = InstantiatedPlotlyNativeChart(
                fig=fig, chart_name=chart_name, default_legend=False)
            new_chart.annotation_upper_left = annotation_upper_left
            new_chart.annotation_upper_right = annotation_upper_right
            new_chart.set_csv_data_from_dataframe(
                data_df)
        return new_chart
    
    def generate_bar_chart_by_category(self, data_df, column_with_categories, x_axis_column, y_axis_column_list, x_axis_title='', y_axis_title='', chart_name='', ticksuffix='', annotation_upper_left={}, annotation_upper_right={}):
        '''
        data_df : dataframe with data to plot but these data are repeated as many times as number of categories
        x_axis_column : string column name of x axis
        y_axis_column_list : list columns names for y bar to plot  
        '''
        
        categories_list=data_df[column_with_categories].unique()
        # Create figure
        fig = go.Figure()
        for category in categories_list:
            cf_df_by_cat=data_df[data_df[column_with_categories]==category]
            for column in y_axis_column_list:
                if column in data_df:
                    fig.add_trace(
                        go.Bar(
                            x=cf_df_by_cat[x_axis_column].values.tolist(),
                            y=cf_df_by_cat[column].values.tolist(),
                            name=f'{category}',
                            textposition='inside',
                            xaxis='x',
                            yaxis='y',
                            visible=True,
                        )
                    )

        fig.update_layout(
            autosize=True,
            xaxis=dict(
                title=x_axis_title,
                titlefont_size=12,
                tickfont_size=10,
                automargin=True
            ),
            yaxis=dict(
                title=y_axis_title,
                titlefont_size=12,
                tickfont_size=10,
                ticksuffix=f'{ticksuffix}',
                automargin=True,

            ),
        )

        new_chart = None
        if len(fig.data):
            # Create native plotly chart
            new_chart = InstantiatedPlotlyNativeChart(
                fig=fig, chart_name=chart_name, default_legend=False)
            new_chart.annotation_upper_left = annotation_upper_left
            new_chart.annotation_upper_right = annotation_upper_right
            new_chart.set_csv_data_from_dataframe(
                data_df)
        return new_chart

import plotly.graph_objects as go
from sos_trades_core.tools.post_processing.post_processing_tools import (
    align_two_y_axes,
    format_currency_legend,
)
from sos_trades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import (
    InstantiatedPlotlyNativeChart,
)
import pandas as pd
import numpy as np
import math


class CommonCharts(InstantiatedPlotlyNativeChart):

    """Class to host standard post post processing charts templates"""

    def __init__(self):

        super().__init__(go.Figure())
        self.default_chart = InstantiatedPlotlyNativeChart(go.Figure())
        self.default_legend = self.default_chart.get_default_legend_layout()

    def generate_bar_chart(
        self,
        data_df: pd.DataFrame,
        x_axis_column: str,
        y_axis_column_list: list,
        x_axis_title: str = '',
        y_axis_title: str = '',
        chart_name: str = '',
        ticksuffix: str = '',
        annotation_upper_left: dict = {},
        annotation_upper_right: dict = {},
        labels_dict: dict = {},
        annotations: list = [],
        updatemenus: list = [],
        barmode: str = 'stack',
        text_inside_bar: bool = False,
        add_cumulated: bool = False,
        column_val_cum_sum: str = None,
        showlegend: bool = True,
    ) -> InstantiatedPlotlyNativeChart:
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
                        name=f'{labels_dict.get(col,col)}'
                        if len(labels_dict) != 0
                        else col,
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
                    name=f'Cumulative {labels_dict.get(column_val_cum_sum,column_val_cum_sum)}'
                    if len(labels_dict) != 0
                    else f'Cumulative {column_val_cum_sum}',
                    xaxis='x',
                    yaxis='y2',
                    visible=True,
                    mode='lines',
                )
            )

        fig.update_layout(
            xaxis=dict(
                title=x_axis_title, titlefont_size=12, tickfont_size=10, automargin=True
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
                fig=fig, chart_name=chart_name, default_legend=False
            )
            new_chart.annotation_upper_left = annotation_upper_left
            new_chart.annotation_upper_right = annotation_upper_right
            new_chart.set_csv_data_from_dataframe(data_df)
        return new_chart

    def generate_lines_chart(
        self,
        data_df: pd.DataFrame,
        x_axis_column: str,
        y_axis_column_list: list,
        x_axis_title: str = '',
        y_axis_title: str = '',
        chart_name: str = '',
        ticksuffix: str = '',
        annotation_upper_left: dict = {},
        annotation_upper_right: dict = {},
        labels_dict: dict = {},
        mode: str = 'lines',
        textposition: str = "top center",
    ) -> InstantiatedPlotlyNativeChart:
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
                        name=f'{labels_dict.get(column,column)}'
                        if len(labels_dict) != 0
                        else column,
                        xaxis='x',
                        yaxis='y',
                        visible=True,
                        mode=mode,
                        text=data_df[column].values.tolist(),
                        textposition=textposition,
                    )
                )

        fig.update_layout(
            autosize=True,
            xaxis=dict(
                title=x_axis_title, titlefont_size=12, tickfont_size=10, automargin=True
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
                fig=fig, chart_name=chart_name, default_legend=False
            )
            new_chart.annotation_upper_left = annotation_upper_left
            new_chart.annotation_upper_right = annotation_upper_right
            new_chart.set_csv_data_from_dataframe(data_df)
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
                fig=fig, chart_name=chart_name, default_legend=False
            )
        return new_chart

    def generate_sunburst_chart(
        self,
        sunburst_labels,
        sunburst_parents,
        sunburst_values,
        sunburst_text,
        chart_name='',
        branchvalues="total",
        textinfo='label+text',
        # hoverinfo='label+text+percent parent',
        hoverinfo=None,
    ):

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
        new_chart = InstantiatedPlotlyNativeChart(fig=fig, chart_name=chart_name)

        return new_chart

    def generate_pie(self, df, lab_column_name, val_column_name, title):
        fig = go.Figure()

        if (lab_column_name in df) & (val_column_name in df):

            df.sort_values(by=val_column_name, axis=0, ascending=False, inplace=True)
            df['percent'] = df[val_column_name] / df[val_column_name].sum()
            df_to_show = df.loc[df['percent'] > 0.01]

            other_value = 0
            pie_labels = df_to_show[lab_column_name].values.tolist()
            pie_values = df_to_show[val_column_name].values.tolist()

            other_value = df.loc[df['percent'] <= 0.01, val_column_name].sum()
            if other_value > 0:
                pie_labels.append('Other (< 1% each)')
                pie_values.append(other_value)
            pie_text = [f'{round(val,1)}' for val in pie_values]

            fig.add_trace(
                go.Pie(
                    labels=pie_labels,
                    values=pie_values,
                    text=pie_text,
                    hovertext=pie_text,
                    textinfo='label',
                    hoverinfo='label+text+percent',
                    textposition='inside',
                )
            )

        # Chart Layout update
        fig.update_layout(
            margin=dict(t=80, l=15, r=0, b=10),
            legend=dict(orientation='v'),
            showlegend=False,
            uniformtext=dict(minsize=12, mode='hide'),
        )

        if len(fig.data) > 0:
            # Create native plotly chart
            chart_name = title
            new_chart = InstantiatedPlotlyNativeChart(fig=fig, chart_name=chart_name)

            return new_chart

    def generate_lines_chart_by_category(
        self,
        data_df,
        column_with_categories,
        x_axis_column,
        y_axis_column_list,
        x_axis_title='',
        y_axis_title='',
        chart_name='',
        ticksuffix='',
        annotation_upper_left={},
        annotation_upper_right={},
        mode='lines',
        textposition="top center",
        fill=None,
        stackgroup=None,
        hoveron=None,
        legend=None,
        name=None,
        fillcolor=None,
        line=None,
        legendgroup=None,
        add_cumulated=False,
        marker=dict(size=12,),
        string_text=False,
        
    ):
        '''
        data_df : dataframe with data to plot but these data are repeated as many times as number of categories
        x_axis_column : string column name of x axis
        y_axis_column_list : list columns names for y bar to plot
        add_cumulated : True to add the cumulative serie of the data
        if mode = 'markers' or mode = 'markers + text'
            marker= {'color': str, 'size': integer, }
        '''

        categories_list = data_df[column_with_categories].unique()
        # Create figure
        fig = go.Figure()
        for category in categories_list:
            cf_df_by_cat = data_df[data_df[column_with_categories] == category]
            for column in y_axis_column_list:
                if column in data_df:
                    fig.add_trace(
                        go.Scatter(
                            x=cf_df_by_cat[x_axis_column].values.tolist(),
                            y=cf_df_by_cat[column].cumsum().values.tolist()
                            if add_cumulated
                            else cf_df_by_cat[column].values.tolist(),
                            name=name if name is not None else f'{category}',
                            xaxis='x',
                            yaxis='y',
                            visible=True,
                            mode=mode,
                            fill=fill,
                            fillcolor=fillcolor,
                            stackgroup=stackgroup,
                            hoveron=hoveron,
                            line=line,
                            text=f'{category}' if string_text else cf_df_by_cat[column].values.tolist() ,
                            marker= marker,
                            textposition=textposition,
                            legendgroup=legendgroup,
                        )
                    )

        fig.update_layout(
            autosize=True,
            xaxis=dict(
                title=x_axis_title, titlefont_size=12, tickfont_size=10, automargin=True
            ),
            yaxis=dict(
                title=y_axis_title,
                titlefont_size=12,
                tickfont_size=10,
                ticksuffix=f'{ticksuffix}',
                automargin=True,
            ),
            legend=None,
        )

        new_chart = None
        if len(fig.data):
            # Create native plotly chart
            new_chart = InstantiatedPlotlyNativeChart(
                fig=fig, chart_name=chart_name, default_legend=False
            )
            new_chart.annotation_upper_left = annotation_upper_left
            new_chart.annotation_upper_right = annotation_upper_right
            new_chart.set_csv_data_from_dataframe(data_df)
        return new_chart

    def generate_bar_chart_by_category(
        self,
        data_df,
        column_with_categories,
        x_axis_column,
        y_axis_column_list,
        x_axis_title='',
        y_axis_title='',
        chart_name='',
        ticksuffix='',
        annotation_upper_left={},
        annotation_upper_right={},
        compute_legend_title=False,
        compute_colors_details=False,
        barmode='stack',
    ):
        '''
        data_df : dataframe with data to plot but these data are repeated as many times as number of categories
        x_axis_column : string column name of x axis
        y_axis_column_list : list columns names for y bar to plot
        '''

        categories_list = data_df[column_with_categories].unique()
        # Create figure
        fig = go.Figure()

        # compute legend if needed
        if compute_legend_title is True:
            categories_dict = dict.fromkeys(categories_list)
            start = 1
            for index, category in enumerate(categories_list):
                if index < len(categories_list) - 1:
                    categories_dict[category] = f'{start} to {category}'
                    start = category + 1
                else:
                    categories_dict[category] = f'> {categories_list[index-1]}'

        # if compute_colors_details is True:
        #     color_scale_ref = px.colors.qualitative.D3
        #     color_scale = [px.colors.hex_to_rgb(
        #         color_hex) for color_hex in color_scale_ref]

        for category in categories_list:
            name = f'{category}'
            if compute_legend_title is True:
                name = f'{categories_dict[category]} Aircrafts'

            cf_df_by_cat = data_df[data_df[column_with_categories] == category]
            for column in y_axis_column_list:
                if column in data_df:
                    fig.add_trace(
                        go.Bar(
                            x=cf_df_by_cat[x_axis_column].values.tolist(),
                            y=cf_df_by_cat[column].values.tolist(),
                            name=name,
                            textposition='inside',
                            xaxis='x',
                            yaxis='y',
                            visible=True,
                        )
                    )

        fig.update_layout(
            autosize=True,
            xaxis=dict(
                title=x_axis_title, titlefont_size=12, tickfont_size=10, automargin=True
            ),
            yaxis=dict(
                title=y_axis_title,
                titlefont_size=12,
                tickfont_size=10,
                ticksuffix=f'{ticksuffix}',
                automargin=True,
            ),
            barmode=barmode,
        )

        new_chart = None
        if len(fig.data):
            # Create native plotly chart
            new_chart = InstantiatedPlotlyNativeChart(
                fig=fig, chart_name=chart_name, default_legend=False
            )
            new_chart.annotation_upper_left = annotation_upper_left
            new_chart.annotation_upper_right = annotation_upper_right
            new_chart.set_csv_data_from_dataframe(data_df)
        return new_chart

    def generate_bar_line_chart_by_category(
        self,
        data_df_bar,
        data_df_line,
        column_with_categories,
        x_axis_column,
        y_bar_axis_column_list,
        y_line_axis_column_list,
        xaxis='x',
        yaxis_bar='y',
        yaxis_line='y2',
        colors_dict={},
        legends_dict={},
        x_axis_title='',
        y_axis_title_bar='',
        y_axis_title_line='',
        chart_name='',
        ticksuffix='',
        annotation_upper_left={},
        annotation_upper_right={},
        mode='lines',
        barmode='stack',
        line=None,
    ):

        categories_list = data_df_bar[column_with_categories].unique()
        # Create figure
        fig = go.Figure()
        for category in categories_list:
            cf_df_by_cat = data_df_bar[data_df_bar[column_with_categories] == category]
            for column in y_bar_axis_column_list:
                if column in data_df_bar:
                    x_values = cf_df_by_cat[x_axis_column].values
                    color = (
                        [colors_dict[column]] * len(x_values)
                        if len(colors_dict) != 0
                        else None
                    )
                    fig.add_trace(
                        go.Bar(
                            x=x_values.tolist(),
                            y=cf_df_by_cat[column].values.tolist(),
                            name=legends_dict[column]
                            if column in legends_dict
                            else f'{category}',
                            xaxis=xaxis,
                            yaxis=yaxis_bar,
                            visible=True,
                            marker=dict(color=color),
                        )
                    )

        categories_list = data_df_line[column_with_categories].unique()
        # Create figure
        for category in categories_list:
            cf_df_by_cat = data_df_line[
                data_df_line[column_with_categories] == category
            ]
            for column in y_line_axis_column_list:
                if column in data_df_line:
                    fig.add_trace(
                        go.Scatter(
                            x=cf_df_by_cat[x_axis_column].values.tolist(),
                            y=cf_df_by_cat[column].values.tolist(),
                            name=legends_dict[column]
                            if column in legends_dict
                            else f'{category}',
                            xaxis=xaxis,
                            yaxis=yaxis_line,
                            visible=True,
                            mode=mode,
                            text=cf_df_by_cat[column].values.tolist(),
                            line=line,
                        )
                    )

        fig.update_layout(
            autosize=True,
            xaxis=dict(
                title=x_axis_title, titlefont_size=12, tickfont_size=10, automargin=True
            ),
            yaxis=dict(
                title=y_axis_title_bar,
                titlefont_size=12,
                tickfont_size=10,
                ticksuffix=f'{ticksuffix}',
                automargin=True,
            ),
            yaxis2=dict(
                title=y_axis_title_line,
                titlefont_size=12,
                tickfont_size=10,
                ticksuffix=f'{ticksuffix}',
                automargin=True,
                anchor="x",
                overlaying="y",
                side="right",
            ),
            barmode=barmode,
        )

        new_chart = None
        if len(fig.data):
            # Create native plotly chart
            fig = align_two_y_axes(fig, GRIDLINES=4)
            new_chart = InstantiatedPlotlyNativeChart(
                fig=fig, chart_name=chart_name, default_legend=False
            )
            new_chart.annotation_upper_left = annotation_upper_left
            new_chart.annotation_upper_right = annotation_upper_right
        return new_chart

    def generate_bar_chart_slider(
        self,
        data_df,
        column_with_categories,
        x_axis_column,
        y_axis_column,
        x_axis_title='',
        y_axis_title='',
        chart_name='',
        ticksuffix='',
        textposition='inside',
        annotation_upper_left={},
        annotation_upper_right={},
    ):

        categories_list = data_df[column_with_categories].unique()
        x_axis_list = data_df[x_axis_column].unique()
        scenario_list = ['Scenario']

        # Create figure
        fig = go.Figure()
        annotations_year = []
        for y in range(len(x_axis_list)):
            if y == len(x_axis_list) - 1:
                visible = True
            else:
                visible = False

            # compute data of the year y for each category
            df_dic = {}
            for category in categories_list:
                df_dic[category] = [
                    data_df.loc[
                        data_df[column_with_categories] == category, y_axis_column
                    ].values[y]
                ]

            dic = {}
            for j in range(len(scenario_list)):
                tot = sum(list(df_dic.values())[i][j] for i in range(len(df_dic)))
                dic[scenario_list[j]] = tot

            for key, fcf in df_dic.items():
                # category bar
                fig.add_trace(
                    go.Bar(
                        x=['  '],
                        y=fcf,
                        name=f'{key}',
                        text=f'{key}',
                        textposition=textposition,
                        xaxis='x',
                        yaxis='y',
                        visible=visible,
                    )
                )

            annotations = []
            if len(df_dic) > 0:
                # Create annotations
                count = 0
                for scen, total in dic.items():
                    annotation = dict(
                        x=count,
                        yref="y",
                        y=total,
                        text=format_currency_legend(total, ' '),
                        xanchor='auto',
                        yanchor='bottom',
                        showarrow=False,
                        font=dict(
                            family="Arial",
                            size=max(min(16, 100 / len(scenario_list)), 6),
                            color="#000000",
                        ),
                        visible=True,
                    )
                    annotations.append(annotation)
                    count += 1

            annotations_year.append(annotations)

        # Create and add slider
        steps = []
        for i in range(int(len(fig.data) / len(categories_list))):
            step = dict(
                method='update',
                args=[
                    {'visible': [False] * len(fig.data)},
                    {
                        'title': f'Total Cumulated Deliveries in {x_axis_list[i]}',
                        'annotations[0].text': annotations_year[i][0]['text'],
                    },
                ],  # layout attribute
                label=f'{x_axis_list[i]}',
            )
            # Toggle i'th trace to 'visible'
            for k in range(len(categories_list)):
                step['args'][0]['visible'][i * len(categories_list) + k] = True
            steps.append(step)

        sliders = [
            dict(
                active=len(steps) - 1,
                currentvalue={'prefix': 'Select Year, currently: '},
                steps=steps,
            )
        ]

        # Chart Layout update
        fig.update_layout(
            sliders=sliders,
            xaxis=dict(
                automargin=True,
                visible=True,
                # type='multicategory'
            ),
            yaxis=dict(
                title=y_axis_title,
                titlefont=dict(color="#1f77b4"),
                tickfont=dict(color="#1f77b4"),
                automargin=True,
            ),
            updatemenus=[
                dict(
                    buttons=list(
                        [
                            dict(
                                args=[
                                    {'annotations[0].visible': True, 'barmode': 'stack'}
                                ],
                                label="Sum",
                                method="relayout",
                            ),
                            dict(
                                args=[
                                    {
                                        'annotations[0].visible': False,
                                        'barmode': 'group',
                                    }
                                ],
                                label="Compare",
                                method="relayout",
                            ),
                        ]
                    ),
                    direction='down',
                    type='dropdown',
                    pad={"r": 0, "t": 0},
                    showactive=True,
                    active=0,
                    x=1.0,
                    y=1.01,
                    yanchor='bottom',
                    xanchor='right',
                ),
            ],
            showlegend=False,
            barmode='stack',
            # barmode='relative',
            # autosize=True,
            annotations=annotations_year[-1],
            # margin=dict(l=0.25, b=100)
        )

        new_chart = None
        last_year = int(
            data_df.loc[
                data_df[column_with_categories] == categories_list[0], x_axis_column
            ].values[-1]
        )
        if len(fig.data):
            # Create native plotly chart
            chart_name = f'{chart_name} in {last_year}'
            new_chart = InstantiatedPlotlyNativeChart(
                fig=fig, chart_name=chart_name, default_legend=False
            )

        export_data = pd.DataFrame(df_dic)
        export_data['scenario_id'] = scenario_list
        new_chart.set_csv_data_from_dataframe(export_data)

        return new_chart

    def generate_lines_chart_with_display(
        self,
        data_df: pd.DataFrame,
        column_with_categories,
        col_pretty_list: list,
        x_axis_column: str,
        y_axis_column_list: list,
        x_axis_title: str = '',
        y_axis_title: str = '',
        layout: str = '',
        mode:str = 'markers+lines',
        ticksuffix: str = '',
        chart_name: str ='',
        name=None,
        textposition="top center",
        fill=None,
        stackgroup=None,
        hoveron=None,
        legend=None,
        fillcolor=None,
        line=None,
        legendgroup=None,
        add_cumulated:bool =False,
        marker=dict(size=12,),
        string_text:bool =False,
        annotation_upper_left:dict ={},
        annotation_upper_right:dict ={},

    ):        
        '''Generate a bar chart from data in a dataframe

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
            InstantiatedPlotlyNativeChart: Plotly Instanciated chart with data'''
        
        categories_list = data_df[column_with_categories].unique()
        # Create figure
        fig = go.Figure()
        vis=True
        for category in categories_list:
            cf_df_by_cat = data_df[data_df[column_with_categories] == category]
            for column in y_axis_column_list:
                if column in data_df:
                    if column == y_axis_column_list[0]:
                        vis = True
                    else:
                        vis = False                    
                    fig.add_trace(
                        go.Scatter(
                            x=cf_df_by_cat[x_axis_column].values.tolist(),
                            y=(cf_df_by_cat[column].cumsum().values.tolist()
                            if add_cumulated
                            else cf_df_by_cat[column].values.tolist()),
                            name=(name if name is not None else f'{category}'),
                            xaxis='x',
                            yaxis='y',
                            visible=vis,
                            mode=mode,
                            fill=fill,
                            fillcolor=fillcolor,
                            stackgroup=stackgroup,
                            hoveron=hoveron,
                            line=line,
                            text=f'{category}' if string_text else cf_df_by_cat[column].values.tolist() ,
                            marker= marker,
                            textposition=textposition,
                            legendgroup=legendgroup,
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
            legend=None,
        )
        
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list([dict(
                        args=[
                            {'visible': [True if i == j else False for j in range(int(len(fig.data)/len(categories_list)))] * len(categories_list)},
                            {'title': f'<b>{col_pretty_list[i]} for {layout} layout</b>'},
                        ],
                        label=col_pretty_list[i],
                        method="update"
                    ) for i in range(len(col_pretty_list))]),
                    direction='down',
                    type='dropdown',
                    pad={"r": 0, "t": 0},
                    showactive=True,
                    active=0,
                    x=1.0,
                    y=1.01,
                    yanchor='bottom',
                    xanchor='right'
                ),
            ]
        )

        new_chart = None
        if len(fig.data):
            # Create native plotly chart
            chart_name = f'{chart_name}'
            new_chart = InstantiatedPlotlyNativeChart(
                fig=fig, chart_name=chart_name, default_legend=False
            )
            new_chart.annotation_upper_left = annotation_upper_left
            new_chart.annotation_upper_right = annotation_upper_right
            # new_chart.set_csv_data_from_dataframe(data_df)
        
        return new_chart
    
    def generate_marker_chart_by_category_with_isoline(
        self,
        data_df: pd.DataFrame,
        column_with_categories,
        var1_name: str,
        var2_name: str,
        var_to_compare_ref: str,
        mpax_ref: float,
        x_axis_title: str = '',
        y_axis_title: str = '',
        layout: str = '',
        mode:str = 'markers+text',
        ticksuffix: str = '',
        chart_name: str ='',
        name=None,
        textposition="top center",
        marker=dict(size=12,),
        string_text:bool =False,
        annotation_upper_left:dict ={},
        annotation_upper_right:dict ={},
        ):
    
        categories_list = data_df[column_with_categories].unique()
        # Create figure
        fig = go.Figure()
  
        for category in categories_list:
            cf_df_by_cat = data_df[data_df[column_with_categories] == category]
            # for column in y_axis_column_list:
            #     if column in data_df:
                    

            coc_x = []
            cocs_y = []
            mpax_z = []

            mpax = cf_df_by_cat[var_to_compare_ref].to_list()   
            coc=cf_df_by_cat[var1_name].to_list()
            cocs=cf_df_by_cat[var2_name].to_list()

            fig.add_trace(
                go.Scatter(
                    x=coc,
                    y=cocs,
                    name=f'{category}',
                    text=[f'{category}'],
                    textposition='top right',
                    visible=True,
                    mode=mode,
                    marker=marker)
            )

            coc_x += coc
            cocs_y += cocs
            mpax_z += mpax

        # create isolines from min -10 to max + 10, with step 10
        list_seat = [*range(math.trunc(min(mpax_z) / 10) * 10 - 10,
                            math.ceil(max(mpax_z) / 10) * 10 + 20, 10)]

        # select plot window
        range_x = [min(coc_x) - 10, max(coc_x) + 10]
        range_y = [min(cocs_y) - 10, max(cocs_y) + 10]

        # add text isoline
        min_x_text = range_x[0] / 100 + 5 / \
            100 * (range_x[1] - range_x[0]) / 100
        min_y_text = range_y[0] / 100 + 5 / \
            100 * (range_y[1] - range_y[0]) / 100

        fig.add_shape(type='line', x0=0, x1=0,  y0=range_y[0], y1=range_y[1],
                      line=dict(color='black'))

        fig.add_shape(type='line', x0=range_x[0], x1=range_x[1], y0=0, y1=0,
                      line=dict(color='black'))

        for i in list_seat:
            # isoline : y = m*x + p
            m_i = mpax_ref / i
            p_i = (mpax_ref - i) / i
            x0 = range_x[0] / 100
            x1 = range_x[1] / 100
            y0 = (m_i * x0 + p_i)
            y1 = (m_i * x1 + p_i)

            # text on isoline
            x_text = min_x_text
            y_text = m_i * min_x_text + p_i

            if min_y_text > y_text:
                y_text = min_y_text
                x_text = (min_y_text - p_i) / m_i

            fig.add_shape(type='line', opacity=0.1,
                          x0=x0 * 100, x1=x1 * 100,
                          y0=y0 * 100, y1=y1 * 100, )

            fig.add_annotation(x=x_text * 100,
                               y=y_text * 100,
                               text=str(i) + ' pax',
                               textangle=-20,
                               showarrow=False,
                               opacity=0.6)

        fig.update_layout(
            autosize=True,
            xaxis=dict(
                range=range_x,
                title=x_axis_title,
                titlefont_size=12,
                tickfont_size=10,
                automargin=True,
                ticksuffix=ticksuffix,
                dtick=5,
                tick0=0
            ),
            yaxis=dict(
                range=range_y,
                title=y_axis_title,
                titlefont_size=12,
                tickfont_size=10,
                automargin=True,
                ticksuffix=ticksuffix,
                dtick=5,
                tick0=0
            ),
            legend=self.default_legend,
        )

        new_chart = None
        if len(fig.data):
            # Create native plotly chart
            new_chart = InstantiatedPlotlyNativeChart(
                fig=fig, chart_name=chart_name, default_legend=False)

        return new_chart
    

'''
Copyright 2022 Airbus SAS
Modifications on 2023/02/23-2023/11/03 Copyright 2023 Capgemini

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
from copy import deepcopy
import plotly.graph_objects as go
from sostrades_core.tools.post_processing.post_processing_tools import (
    align_two_y_axes,
    format_currency_legend,
)
from sostrades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import (
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
            offsetgroup=None,
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
                        name=f'{labels_dict.get(col, col)}'
                        if len(labels_dict) != 0
                        else col,
                        text=col if text_inside_bar else '',
                        textposition='inside',
                        xaxis='x',
                        yaxis='y',
                        visible=True,
                        offsetgroup=offsetgroup,
                    )
                )

        if add_cumulated:
            fig.add_trace(
                go.Scatter(
                    x=data_df[x_axis_column].values.tolist(),
                    y=data_df[column_val_cum_sum].cumsum().values.tolist(),
                    name=f'Cumulative {labels_dict.get(column_val_cum_sum, column_val_cum_sum)}'
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
                        name=f'{labels_dict.get(column, column)}'
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

    def generate_columns_infos_table(self, info_df, in_dict=dict()):
        '''
        Function to make more understandable the variables' names
        that appear in post-processing tables
        '''

        info_dict = deepcopy(in_dict)
        columns_info = columns_info = {
            'index': {'label': 'Name', 'format': None},
            'scenario_id': {'label': 'Scenario', 'format': None},
            'irr': {'label': 'Internal Rate of Return (IRR)', 'format': 'percent'},
            'npv': {'label': 'Net Present Value (NPV)', 'format': 'currency'},
            'year_break_even_cashflow': {
                'label': 'Cashflow Breakeven Year',
                'format': None,
            },
            'year_break_even_discounted_cashflow': {
                'label': 'Discounted Cashflow Breakeven Year ',
                'format': None,
            },
            'peak_exposure': {'label': 'Peak Exposure', 'format': 'currency'},
            'total_free_cash_flow': {
                'label': 'Total Free Cashflow',
                'format': 'currency',
            },
        }

        if ('last_year' in info_df) & ('year_start_escalation_nrc' in info_df):
            last_year = int(info_df['last_year'].values[0])
            year_start_escalation_nrc = int(
                info_df['year_start_escalation_nrc'].values[0]
            )

            info_dict['total_cumul_nrc'] = {
                'label': f'Total Cumulative NRC (ec {year_start_escalation_nrc})',
                'format': 'currency',
            }
            info_dict['nrc_last_year'] = {
                'label': f'NRC in {last_year}',
                'format': 'currency',
            }
            del info_df['last_year']
            # del info_df['year_start_escalation_rc']
            del info_df['year_start_escalation_nrc']

        key_info_list = list(info_df.columns)
        for cle in key_info_list:
            if cle.startswith('total_cumul_sales'):
                if cle == 'total_cumul_sales':
                    info_dict['total_cumul_sales'] = {
                        'label': 'Total Cumulative Sales',
                        'format': None,
                    }
                else:
                    ac = cle.split('total_cumul_sales_')[-1]
                    info_dict[f'total_cumul_sales_{ac}'] = {
                        'label': f'Total Cumulative Sales {ac}',
                        'format': None,
                    }

            elif cle.startswith('year_start_escalation_rc'):
                if cle == 'year_start_escalation_rc':
                    year_start_escalation_rc = int(info_df[cle].values[0])
                    info_dict['total_cumul_rc'] = {
                        'label': f'Total Cumulative RC (ec {year_start_escalation_rc})',
                        'format': 'currency',
                    }
                else:
                    ac = cle.split('year_start_escalation_rc_')[-1]
                    year_start_escalation_rc = int(info_df[cle].values[0])
                    info_dict[f'total_cumul_rc_{ac}'] = {
                        'label': f'Total Cumulative RC (ec {year_start_escalation_rc}) for {ac}',
                        'format': 'currency',
                    }

                del info_df[cle]

            elif cle.startswith('rc_last_year'):
                if cle == 'rc_last_year':
                    year_start_escalation_rc = int(info_df[cle].values[0])
                    info_dict['rc_last_year'] = {
                        'label': f'RC in {last_year}',
                        'format': 'currency',
                    }
                else:
                    ac = cle.split('rc_last_year_')[-1]
                    year_start_escalation_rc = int(info_df[cle].values[0])
                    info_dict[f'rc_last_year_{ac}'] = {
                        'label': f'RC in {last_year} for {ac}',
                        'format': 'currency',
                    }

            elif cle.startswith('sale_price_last_year'):
                if cle == 'sale_price_last_year':
                    year_start_escalation_rc = int(info_df[cle].values[0])
                    info_dict['sale_price_last_year'] = {
                        'label': f'Sale Price in {last_year}',
                        'format': 'currency',
                    }
                else:
                    ac = cle.split('sale_price_last_year_')[-1]
                    year_start_escalation_rc = int(info_df[cle].values[0])
                    info_dict[f'sale_price_last_year_{ac}'] = {
                        'label': f'Sale Price in {last_year} for {ac}',
                        'format': 'currency',
                    }

            elif cle.startswith('contribution_margin_last_year'):
                if cle == 'contribution_margin_last_year':
                    info_dict['contribution_margin_last_year'] = {
                        'label': f'Contribution Margin in {last_year}',
                        'format': 'percent',
                    }
                else:
                    ac = cle.split('contribution_margin_last_year_')[-1]
                    info_dict[f'contribution_margin_last_year_{ac}'] = {
                        'label': f'Contribution Margin in {last_year} for {ac}',
                        'format': 'percent',
                    }

            else:
                pass

        if 'mean_contribution_margin_last_year' in info_df:
            info_dict['mean_contribution_margin_last_year'] = {
                'label': f'Average Contribution Margin in {last_year}',
                'format': 'percent',
            }

        columns_info.update(info_dict)
        return columns_info

    def generate_table_chart(
            self,
            data_df,
            chart_name,
            ticksuffix=None,
    ):
        '''
        data_df : dataframe with data to plot
        '''
        # Create figure
        fig = go.Figure()

        infos = deepcopy(data_df)
        columns_info_dict = self.generate_columns_infos_table(infos)
        columns_names = []
        columns_data = []
        for (key, data) in infos.iteritems():
            if key in columns_info_dict:
                columns_names.append(
                    f'<b>{columns_info_dict[key].get("label", key)}</b>'
                )
                if columns_info_dict[key].get('format', None) == 'currency':
                    columns_data.append(
                        data.apply(format_currency_legend, args=(ticksuffix)).values
                    )
                elif columns_info_dict[key].get('format', None) == 'percent':
                    columns_data.append(
                        data.apply(
                            lambda x: "{0:.2f}%".format(x * 100)
                            if not isinstance(x, str)
                            else x
                        ).values
                    )
                else:
                    columns_data.append(data.values)
            else:
                columns_names.append(f'<b>{key}</b>')
                columns_data.append(data.values)

        fig.add_trace(
            go.Table(
                header=dict(
                    values=columns_names,
                    fill_color='midnightblue',
                    align='center',
                    font_color='white',
                ),
                cells=dict(
                    values=columns_data,
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
            # height=data_df.shape[0] * 10,
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

    def generate_pie(
            self,
            df: pd.DataFrame,
            lab_column_name: str,
            val_column_name: str,
            title: str = '',
            annotation_upper_left: dict = {},
            ticksuffix: str = '',
            threshold_to_show: float = 1,
    ):

        fig = go.Figure()
        threshold = threshold_to_show / 100.0

        if (lab_column_name in df) & (val_column_name in df):

            df.sort_values(by=val_column_name, axis=0, ascending=False, inplace=True)
            df['percent'] = df[val_column_name] / df[val_column_name].sum()
            df_to_show = df.loc[df['percent'] >= threshold]

            other_value = 0
            pie_labels = df_to_show[lab_column_name].values.tolist()
            pie_values = df_to_show[val_column_name].values.tolist()

            other_value = df.loc[df['percent'] <= threshold, val_column_name].sum()
            if other_value > 0:
                pie_labels.append(f'Other (< {threshold_to_show} % each)')
                pie_values.append(other_value)
            pie_text = [f'{round(val, 2)} {ticksuffix}' for val in pie_values]

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
            new_chart.annotation_upper_left = annotation_upper_left

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
            marker=dict(
                size=12,
            ),
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
                            text=f'{category}'
                            if string_text
                            else cf_df_by_cat[column].values.tolist(),
                            marker=marker,
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
            annotations: list = [],
            updatemenus: list = [],
            offsetgroup=None,
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
                    categories_dict[category] = f'> {categories_list[index - 1]}'

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
                            offsetgroup=offsetgroup,
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

        if len(annotations) != 0:
            fig.update_layout(annotations=annotations[0])
        if len(updatemenus) != 0:
            fig.update_layout(updatemenus=updatemenus)

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
            title_by_slider='Total Cumulated Deliveries',
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
                        'title': f'<b>{title_by_slider} in {x_axis_list[i]}</b>',
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
                ticksuffix=ticksuffix,
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
            chart_name = f'<b>{chart_name} in {last_year}</b>'
            new_chart = InstantiatedPlotlyNativeChart(
                fig=fig, chart_name=chart_name, default_legend=False
            )

        export_data = pd.DataFrame(df_dic)
        export_data['scenario_id'] = scenario_list
        new_chart.set_csv_data_from_dataframe(export_data)

        return new_chart

    def generate_lines_chart_with_dropdown(
            self,
            data_df: pd.DataFrame,
            column_with_categories,
            col_pretty_list: list,
            x_axis_column: str,
            y_axis_column_list: list,
            x_axis_title: str = '',
            y_axis_title: str = '',
            layout: str = '',
            mode: str = 'markers+lines',
            ticksuffix: str = '',
            chart_name: str = '',
            name=None,
            textposition="top center",
            fill=None,
            stackgroup=None,
            hoveron=None,
            legend=None,
            fillcolor=None,
            line=None,
            legendgroup=None,
            add_cumulated: bool = False,
            marker=dict(
                size=12,
            ),
            string_text: bool = False,
            annotation_upper_left: dict = {},
            annotation_upper_right: dict = {},
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
        vis = True
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
                            y=(
                                cf_df_by_cat[column].cumsum().values.tolist()
                                if add_cumulated
                                else cf_df_by_cat[column].values.tolist()
                            ),
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
                            text=f'{category}'
                            if string_text
                            else cf_df_by_cat[column].values.tolist(),
                            marker=marker,
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

        # Add dropdowns
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list(
                        [
                            dict(
                                args=[
                                    {
                                        'visible': [
                                                       True if i == j else False
                                                       for j in range(
                                                int(
                                                    len(fig.data) / len(categories_list)
                                                )
                                            )
                                                   ]
                                                   * len(categories_list)
                                    },
                                    {
                                        'title': f'<b>{col_pretty_list[i]} for {layout} layout</b>'
                                    },
                                ],
                                label=col_pretty_list[i],
                                method="update",
                            )
                            for i in range(len(col_pretty_list))
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

    def generate_pie_chart_with_dropdown(
            self,
            df: pd.DataFrame,
            lab_column_name: str,
            val_column_name: str,

            top_to_show: list,
            chart_name: str = '',
            ticksuffix: str = '',
            title_prefix: str = '',
            title_suffix: str = '',
    ):

        fig = go.Figure()
        vis = True

        if (len(df) > 0) & (lab_column_name in df) & (val_column_name in df):

            df.sort_values(by=val_column_name, axis=0, ascending=False, inplace=True)
            # other_value = 0
            for top in top_to_show:

                lines_showed = int(top.split(' ')[-1])
                if lines_showed > len(df):
                    lines_showed = len(df)
                if top == top_to_show[0]:
                    vis = True
                else:
                    vis = False

                df_to_show = df.head(lines_showed)
                pie_labels = df_to_show[lab_column_name].values.tolist()
                pie_values = df_to_show[val_column_name].values.tolist()
                pie_text = [f'{round(val, 2)} {ticksuffix}' for val in pie_values]
                fig.add_trace(
                    go.Pie(
                        labels=pie_labels,
                        values=pie_values,
                        text=pie_text,
                        hovertext=pie_text,
                        visible=vis,
                        textinfo='label+percent',
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
        # Add dropdowns
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list(
                        [
                            dict(
                                args=[
                                    {
                                        'visible': [
                                            True if i == j else False
                                            for j in range(int(len(fig.data)))
                                        ]
                                    },
                                    {
                                        'title': f'<b> {title_prefix} of {top_to_show[i]} components <br> {title_suffix} </b>'
                                    },
                                ],
                                label=top_to_show[i],
                                method="update",
                            )
                            for i in range(len(top_to_show))
                        ]
                    ),
                    direction='down',
                    type='dropdown',
                    pad={"r": 10, "t": 0},
                    showactive=True,
                    active=0,
                    x=1.0,
                    y=1.01,
                    yanchor='bottom',
                    xanchor='right',
                ),
            ]
        )

        new_chart = None
        if len(fig.data) > 0:
            # Create native plotly chart
            chart_name = f'{chart_name}'
            new_chart = InstantiatedPlotlyNativeChart(fig=fig, chart_name=chart_name)
            # new_chart.annotation_upper_left = annotation_upper_left

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
            mode: str = 'markers+text',
            ticksuffix: str = '',
            chart_name: str = '',
            marker=dict(
                size=12,
            ),
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
            coc = cf_df_by_cat[var1_name].to_list()
            cocs = cf_df_by_cat[var2_name].to_list()

            fig.add_trace(
                go.Scatter(
                    x=coc,
                    y=cocs,
                    name=f'{category}',
                    text=[f'{category}'],
                    textposition='top right',
                    visible=True,
                    mode=mode,
                    marker=marker,
                )
            )

            coc_x += coc
            cocs_y += cocs
            mpax_z += mpax

        # create isolines from min -10 to max + 10, with step 10
        list_seat = [
            *range(
                math.trunc(min(mpax_z) / 10) * 10 - 10,
                math.ceil(max(mpax_z) / 10) * 10 + 20,
                10,
            )
        ]

        # select plot window
        range_x = [min(coc_x) - 10, max(coc_x) + 10]
        range_y = [min(cocs_y) - 10, max(cocs_y) + 10]

        # add text isoline
        min_x_text = range_x[0] / 100 + 5 / 100 * (range_x[1] - range_x[0]) / 100
        min_y_text = range_y[0] / 100 + 5 / 100 * (range_y[1] - range_y[0]) / 100

        fig.add_shape(
            type='line',
            x0=0,
            x1=0,
            y0=range_y[0],
            y1=range_y[1],
            line=dict(color='black'),
        )

        fig.add_shape(
            type='line',
            x0=range_x[0],
            x1=range_x[1],
            y0=0,
            y1=0,
            line=dict(color='black'),
        )

        for i in list_seat:
            # isoline : y = m*x + p
            m_i = mpax_ref / i
            p_i = (mpax_ref - i) / i
            x0 = range_x[0] / 100
            x1 = range_x[1] / 100
            y0 = m_i * x0 + p_i
            y1 = m_i * x1 + p_i

            # text on isoline
            x_text = min_x_text
            y_text = m_i * min_x_text + p_i

            if min_y_text > y_text:
                y_text = min_y_text
                x_text = (min_y_text - p_i) / m_i

            fig.add_shape(
                type='line',
                opacity=0.1,
                x0=x0 * 100,
                x1=x1 * 100,
                y0=y0 * 100,
                y1=y1 * 100,
            )

            fig.add_annotation(
                x=x_text * 100,
                y=y_text * 100,
                text=str(i) + ' pax',
                textangle=-20,
                showarrow=False,
                opacity=0.6,
            )

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
                tick0=0,
            ),
            yaxis=dict(
                range=range_y,
                title=y_axis_title,
                titlefont_size=12,
                tickfont_size=10,
                automargin=True,
                ticksuffix=ticksuffix,
                dtick=5,
                tick0=0,
            ),
            legend=self.default_legend,
        )

        new_chart = None
        if len(fig.data):
            # Create native plotly chart
            new_chart = InstantiatedPlotlyNativeChart(
                fig=fig, chart_name=chart_name, default_legend=False
            )

        return new_chart

    def generate_waterfall_chart(
            self, measure_dict, values_dict, text_dict, name, currency
    ):
        # Create figure
        fig = go.Figure()
        year_list = list(measure_dict.keys())
        for y in year_list:
            waterfall = go.Waterfall(
                name=f'<b>{name} Year {y}</b>',
                orientation='h',
                measure=measure_dict[y],
                x=list(values_dict[y].values()),
                textposition='auto',
                text=text_dict[y],
                y=list(values_dict[y].keys()),
                connector={
                    "mode": "between",
                    "line": {
                        "width": 2,
                        "color": "rgb(0, 0, 0)",
                        "dash": "solid",
                    },
                },
                visible=False,
            )
            fig.add_trace(waterfall)
        if len(fig.data):
            fig.data[-1].visible = True

            # Create and add slider
            steps = []
            for i in range(len(fig.data)):
                step = dict(
                    method='update',
                    args=[
                        {'visible': [False] * len(fig.data)},
                        {'title': f'{name} Year {year_list[i]}'},
                    ],  # layout attribute
                    label=f'{year_list[i]}',
                )
                # Toggle i'th trace to 'visible'
                step['args'][0]['visible'][i] = True
                steps.append(step)

            sliders = [
                dict(
                    active=len(steps) - 1,
                    currentvalue={'prefix': 'Select Year, currently: '},
                    steps=steps,
                )
            ]

            fig.update_layout(
                sliders=sliders,
                xaxis=dict(ticksuffix=currency, automargin=True),
                yaxis=dict(automargin=True),
                showlegend=False,
                autosize=True,
            )

        new_chart = None
        if len(fig.data):
            # Create native plotly chart
            chart_name = name
            new_chart = InstantiatedPlotlyNativeChart(
                fig=fig, chart_name=chart_name, default_legend=False
            )

        return new_chart

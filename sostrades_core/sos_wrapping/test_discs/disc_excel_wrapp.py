'''
Copyright 2024 Capgemini

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

from os.path import dirname, join

import numpy as np
import pandas as pd
from pycel import ExcelCompiler

from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)


class DiscExcelWrapp(SoSWrapp):
    # ontology information
    _ontology_data = {
        'label': 'Disc Excel wrapp',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-plane fa-fw',
        'version': '',
    }
    _maturity = 'Fake'
    DESC_IN = {
        'df': {'type': 'dataframe', 'dataframe_descriptor': {'days': ('float', None, True),
                                           'quantity': ('float', None, True)}}
    }
    DESC_OUT = {
        'df_out': {'type': 'dataframe'}
    }

    FILENAME = 'excel_example.xlsx'
    SHEETNAME = 'pricing with team'

    def run(self):

        df = self.get_sosdisc_inputs('df')

        self.filename = join(dirname(__file__), self.FILENAME)

        # Write DataFrame to Excel
        self.write_dataframe(df, self.SHEETNAME, start_row=13, start_col=2)

        # Get results of formulas
        result_df = self.evaluate_column_in_excel(self.SHEETNAME, start_row=13, start_col=4, end_row=28,
                                                  col_name='days total')

        self.store_sos_outputs_values({'df_out': result_df})

    def write_dataframe(self, df, sheet_name, start_row=0, start_col=0):
        """
        Write a pandas DataFrame to an Excel sheet.

        :param df: DataFrame to write to Excel.
        :param sheet_name: Name of the sheet to write to.
        :param start_row: Starting row in the sheet (0-indexed).
        :param start_col: Starting column in the sheet (0-indexed).
        """

        with pd.ExcelWriter(self.filename, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, startcol=start_col, index=False,
                        header=False)

    def evaluate_column_in_excel(self, sheet_name, start_row=0, start_col=0, end_row=None, col_name='result'):
        """
        Get the results of formulas from an Excel sheet.

        :param sheet_name: Name of the sheet to read from.
        :param start_row: Starting row in the sheet (0-indexed).
        :param start_col: Starting column in the sheet (0-indexed).
        :param end_row: Ending row in the sheet (0-indexed).
        :param end_col: Ending column in the sheet (0-indexed).
        :return: DataFrame with the results of the formulas.
        """

        # Use Pycel to evaluate formulas
        comp = ExcelCompiler(filename=self.filename)

        # Create a DataFrame to hold the data
        data = []
        for row in range(start_row, end_row):
            cell = f'{chr(65 + start_col)}{row + 1}'
            value = comp.evaluate(f'{sheet_name}!{cell}')
            data.append(value)

        # df = pd.read_excel(self.filename, sheet_name=sheet_name, header=None)
        # df_results = df.iloc[start_row:end_row, start_col:start_col + 1]

        # df_results.columns = [col_name]
        df = pd.DataFrame({col_name: data})
        return df

    def get_chart_filter_list(self):

        chart_filters = []

        chart_list = ['df out']

        chart_filters.append(ChartFilter(
            'Charts', chart_list, chart_list, 'graphs'))

        return chart_filters

    def get_post_processing_list(self, filters=None):

        instanciated_charts = []
        charts_list = []

        # Overload default value with chart filter
        if filters is not None:
            for chart_filter in filters:
                if chart_filter.filter_key == 'graphs':
                    charts_list = chart_filter.selected_values

        if 'df out' in charts_list:
            chart_name = 'days total'

            y = self.get_sosdisc_outputs('df_out')['days total'].values.tolist()

            x = np.arange(len(y)).tolist()

            new_chart = TwoAxesInstanciatedChart('x (-)', 'y (-)',
                                                 chart_name=chart_name)
            serie = InstanciatedSeries(
                [x], [y], '', 'bar')

            new_chart.series.append(serie)

            instanciated_charts.append(new_chart)

        return instanciated_charts

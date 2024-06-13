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

import pandas as pd
import xlwings as xw
import numpy as np
from os.path import dirname,join

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
        'df': {'type': 'dataframe','dataframe_descriptor': {'days': ('float', None, True),
                                           'quantity': ('float', None, True)}}
    }
    DESC_OUT = {
        'df_out': {'type': 'dataframe'}
    }

    FILENAME ='excel_example.xlsx'
    SHEETNAME = 'pricing with team'

    def run(self):

        df= self.get_sosdisc_inputs('df')

        # Initialize the wrapper
        filename = join(dirname(__file__),self.FILENAME)
        self.init_excel(filename)

        # Write DataFrame to Excel
        self.write_dataframe(df, self.SHEETNAME, start_row=14, start_col=3, end_row=28, end_col=4)

        # Get results of formulas
        days_total_list = self.get_formula_results(self.SHEETNAME, start_row=14, start_col=5, end_row=28, end_col=5)

        result_df=pd.DataFrame({'days total':days_total_list})
        # Save and close the workbook
        self.close_excel()

        self.store_sos_outputs_values({'df_out':result_df})

    def init_excel(self, filename):
        self.filename = filename
        self.app = xw.App(visible=False)  # Run Excel in the background
        self.wb = self.app.books.open(filename)


    def write_dataframe(self, df, sheet_name, start_row=0, start_col=0,end_row=None,end_col=None):
        """
        Write a pandas DataFrame to an Excel sheet.

        :param df: DataFrame to write to Excel.
        :param sheet_name: Name of the sheet to write to.
        :param start_row: Starting row in the sheet (0-indexed).
        :param start_col: Starting column in the sheet (0-indexed).
        """
        sheet = self.wb.sheets[sheet_name]
        if end_row is not None and end_col is not None:
            sheet.range((start_row , start_col ), (end_row , end_col)).value = df.values
        else:
            sheet.range((start_row , start_col )).value = df.values




    def get_formula_results(self, sheet_name, start_row=0, start_col=0, end_row=None, end_col=None):
        """
        Get the results of formulas from an Excel sheet.

        :param sheet_name: Name of the sheet to read from.
        :param start_row: Starting row in the sheet (0-indexed).
        :param start_col: Starting column in the sheet (0-indexed).
        :param end_row: Ending row in the sheet (0-indexed).
        :param end_col: Ending column in the sheet (0-indexed).
        :return: DataFrame with the results of the formulas.
        """
        sheet = self.wb.sheets[sheet_name]
        if end_row is not None and end_col is not None:
            data = sheet.range((start_row , start_col ), (end_row , end_col)).value
        else:
            data = sheet.range((start_row , start_col )).expand().value
        return data


    def close_excel(self):
        """
        Save the workbook and close the Excel application.
        """
        self.wb.close()
        self.app.quit()

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

'''
Copyright 2022 Airbus SAS
Modifications on 29/02/2024-2024/05/16 Copyright 2024 Capgemini

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

import pandas as pd
import plotly.graph_objects as go

from sostrades_core.tools.post_processing.post_processing_plotly_tooling import (
    AbstractPostProcessingPlotlyTooling,
)
from sostrades_core.tools.post_processing.post_processing_tools import (
    escape_str_with_comma,
)

"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
Class that define a table display as post post processing
"""


class InstanciatedTableException(Exception):
    """ Overload Exception basic type 
    """


class InstanciatedTable(AbstractPostProcessingPlotlyTooling):
    """ Class that define a table display as post post processing 
    """

    TABLE_NAME = 'table_name'
    HEADER = 'header'
    CELLS = 'cells'
    STYLES = 'styles'

    def __init__(self, table_name='', header=[], cells=[], styles={}):
        """ Create a new table

        @param table_name : string that contains table name
        @type str

        @param header : string list that contains all table columns name
        @type list of string

        @param cells : string list of list that contains each data line for the table
        type list of string/numeric

        @param styles : dictionary of style to use to display the table
        @type dictionary
        """

        super().__init__()

        # Set the table name list
        self.table_name = table_name

        # table data
        if not isinstance(header, list):
            message = f'"header" argument is intended to be a list not {type(header)}'
            raise TypeError(message)
        self.header = deepcopy(header)

        if not isinstance(cells, list):
            message = f'"cells" argument is intended to be a list not {type(cells)}'
            raise TypeError(message)
        self.cells = deepcopy(cells)

        if not isinstance(styles, dict):
            message = f'"styles" argument is intended to be a list not {type(styles)}'
            raise TypeError(message)
        self.styles = styles

    def __repr__(self):
        """ Overload of the class representation

        @return str, string representation of the instance
        """

        table_string = f'\nname: {self.table_name} \n'
        table_string += f'header: {self.header} \n'
        table_string += f'cells: {self.cells} \n'
        table_string += f'styles: {self.styles} \n'
        table_string += f'annotations\n{super().__repr__()}'

        return table_string

    def to_plotly(self, logger=None):
        """ Convert current instance into a plotly object

        @param logger: logging object to log message
        @type Logging.logger

        @return plotly.graph_objects.go instance
        """

        default_font_color = 'black'
        default_background_color = 'white'
        row_colors = []
        font_color = []

        number_of_rows = 0

        if len(self.cells) > 0:
            number_of_rows = len(self.cells[0])

            for index in range(number_of_rows):

                # Set first column in bold
                self.cells[0][index] = f'<b>{self.cells[0][index]}</b>'

                if index in self.styles:
                    row_colors.append(self.styles[index].background_color)
                    font_color.append(self.styles[index].font_color)
                else:
                    row_colors.append(default_background_color)
                    font_color.append(default_font_color)

            header = {}
            header.update(
                {'values': [f'<b>{header_value}</b>' for header_value in self.header]})
            header.update({'align': 'center'})
            header.update({'line': {'width': 1, 'color': 'black'}})
            header.update({'fill': {'color': 'grey'}})
            header.update({'font': {'family': 'Arial', 'size': 12,
                                    'color': 'white'}})

            cells = {}
            cells.update({'values': self.cells})
            cells.update({'align': 'center'})
            cells.update({'line': {'color': 'black', 'width': 1}})
            cells.update({'fill': {'color': [row_colors]}})
            cells.update(
                {'font': {'family': 'Arial', 'size': 11, 'color': [font_color]}})

            # -- Annotations management
            chart_annotations = []
            # Upper left annotations
            upper_left_annotations = self.get_default_annotations_upper_left_layout()
            if len(upper_left_annotations.keys()) > 0:
                chart_annotations.append(upper_left_annotations)

            # Upper right annotations
            upper_right_annotations = self.get_default_annotations_upper_right_layout()
            if len(upper_right_annotations.keys()) > 0:
                chart_annotations.append(upper_right_annotations)

            table = go.Table(header=header, cells=cells)

            fig = go.Figure(data=[table])

            fig.update_layout(
                title_text=self.table_name,
                plot_bgcolor='rgba(228, 222, 249, 0.65)',
                showlegend=False,
                autosize=True,
                height=number_of_rows * 30 + 250,
                annotations=chart_annotations)

            return fig

    @staticmethod
    def from_pd_df(table_name:str, df:pd.DataFrame):
        return InstanciatedTable(table_name=table_name, header=df.columns.to_list(), cells=df.values.T.tolist())

    def __to_csv(self):
        header_text_list = [str(hd).replace('<b>', '').replace(
            '</b>', '') for hd in self.header]
        # Escaping all comma
        header_text_list = [escape_str_with_comma(
            ht) for ht in header_text_list]
        csv_list = [','.join(header_text_list)]

        if len(self.cells) > 0:
            for cell in self.cells:
                current_cell_text_list = [str(cl).replace(
                    '<b>', '').replace('</b>', '') for cl in cell]
                # Escaping all comma
                current_cell_text_list = [escape_str_with_comma(
                    cl) for cl in current_cell_text_list]
                csv_list.append(','.join(current_cell_text_list))

        self.set_csv_data(csv_list)

    def to_plotly_dict(self, logger=None):
        """ Method that convert current instance to plotly object and then to a dictionary

        @param logger: logger instance
        @type Logging.loger
        """
        json = self.to_plotly(logger).to_dict()

        if self._plot_csv_data is None:
            self.__to_csv()

        json[self.CSV_DATA] = self._plot_csv_data
        json[self.LOGO_NOTOFFICIAL] = self.logo_notofficial
        json[self.LOGO_OFFICIAL] = self.logo_official
        json[self.LOGO_WORK_IN_PROGRESS] = self.logo_work_in_progress

        return json

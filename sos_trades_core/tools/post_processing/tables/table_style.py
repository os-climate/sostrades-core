'''
Copyright 2022 Airbus SAS

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

"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
Class that define styles than can be applied to a table row
"""

class TableStylesException(Exception):
    """ Overload Exception basic type 
    """


class TableStyles:
    """ Class that define styles than can be applied to a table row
    """

    BACKGROUND_COLOR = 'background_color'
    FONT_COLOR = 'font_color'

    def __init__(self, background_color='white', font_color='black'):
        """ Create a new table style

        :params: background_color : string with a color name for a row background color
        :params: font_color : string with a color name for a row fontregarde color
        :params: cells : string list of list that contains each data line for the table
        """

        self.background_color = background_color
        self.font_color = font_color

    def __repr__(self):
        """ Overload of the class representation
        """

        table_string = f'\nbackground_color: {self.background_color} \n'
        table_string += f'font_color: {self.font_color}'

        return table_string

    def to_dict(self):

        dict_obj = {}
        # Serialize table background color attribute
        dict_obj.update({TableStyles.BACKGROUND_COLOR: self.background_color})

        # Serialize table font color attribute
        dict_obj.update(
            {TableStyles.FONT_COLOR: self.font_color})

        return dict_obj

    def from_dict(self, dict_obj):
        """ Method that initialize from dict the InstanciatedTable class
        """
        # Serialize table background colo attribute
        self.background_color = dict_obj[TableStyles.BACKGROUND_COLOR]

        # Serialize table font color attribute
        self.font_color = dict_obj[TableStyles.FONT_COLOR]

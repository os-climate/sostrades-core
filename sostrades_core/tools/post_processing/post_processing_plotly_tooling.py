'''
Copyright 2022 Airbus SAS
Modifications on 2024/05/16 Copyright 2024 Capgemini

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

from abc import ABC

"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
Class that define annotations for post processing
"""

class AbstractPostProcessingPlotlyTooling(ABC):
    """ Class that define general tools for post processing with plotly
    """
    CSV_DATA = 'csv_data'
    ANNOTATION_UPPER_LEFT = 'annotation_upper_left'
    ANNOTATION_UPPER_RIGHT = 'annotation_upper_right'
    LOGO_OFFICIAL = 'logo_official'
    LOGO_NOTOFFICIAL = 'logo_notofficial'
    LOGO_WORK_IN_PROGRESS = 'logo_work_in_progress'

    def __init__(self):
        """ Initialize members variables
        """

        # Initialize annotation properties
        self.annotation_upper_left = {}
        self.annotation_upper_right = {}
        self._plot_csv_data = None
        self.logo_official = False
        self.logo_notofficial = False
        self.logo_work_in_progress = False

    def add_annotation(self, place_holder, annotation_key, annotation_value):
        """ Add an annotation to the current Chart instance

        :params: place_holder, annotation position
        :type: TwoAxesChartTemplate.ANNOTATION_UPPER_LEFT/TwoAxesChartTemplate.ANNOTATION_UPPER_RIGHT

        :params: annotation_key, annotation key use to make the title
        :type: str

        :params: annotation_value, annotation label to display
        :type: str
        """

        if not isinstance(annotation_key, str):
            raise ValueError('annotation_key parameter must be an str type')

        if not isinstance(annotation_value, str):
            raise ValueError('annotation_value parameter must be an str type')

        if place_holder == AbstractPostProcessingPlotlyTooling.ANNOTATION_UPPER_LEFT:
            self.annotation_upper_left[annotation_key] = annotation_value
        elif place_holder == AbstractPostProcessingPlotlyTooling.ANNOTATION_UPPER_RIGHT:
            self.annotation_upper_right[annotation_key] = annotation_value
        else:
            raise ValueError(
                f'Annotation place_holder parameter should be ont of the following value : {AbstractPostProcessingPlotlyTooling.ANNOTATION_UPPER_LEFT} '
                f'or {AbstractPostProcessingPlotlyTooling.ANNOTATION_UPPER_RIGHT}')

    def add_watermark(self, logo_official=False, logo_notofficial=False, logo_work_in_progress=False):
        """ Add a watermark to the current Chart instance

        :params: logo_official, image "official" for the watermark to add
        :type: boolean

        :params: logo_non_official, image "non official" for the watermark to add
        :type: boolean

        :params: logo_work_in_progress, image "work in progress" for the watermark to add
        :type: boolean
        """

        if not isinstance(logo_notofficial, bool):
            raise ValueError('logo_official parameter must be a boolean')
        if not isinstance(logo_official, bool):
            raise ValueError('logo_non_official parameter must be a boolean')
        if not isinstance(logo_work_in_progress, bool):
            raise ValueError(
                'logo_work_in_progress parameter must be a boolean')

        if logo_notofficial:
            self.logo_notofficial = logo_notofficial
        if logo_official:
            self.logo_official = logo_official
        if logo_work_in_progress:
            self.logo_work_in_progress = logo_work_in_progress

    def __repr__(self):
        """ Overload of the class representation

            Allow to hide password_hash from serializer point of view
        """

        chart_string = [f'anno. left: {self.annotation_upper_left}',
                        f'anno. right: {self.annotation_upper_right}',
                        ]

        return '\n'.join(chart_string)

    def to_dict(self):
        """ Convert current instance to disctonary object
        """
        dict_obj = {}

        # Serialize annotation upper left attribute
        dict_obj.update(
            {AbstractPostProcessingPlotlyTooling.ANNOTATION_UPPER_LEFT: self.annotation_upper_left})

        # Serialize annotation upper right attribute
        dict_obj.update(
            {AbstractPostProcessingPlotlyTooling.ANNOTATION_UPPER_RIGHT: self.annotation_upper_right})

        return dict_obj

    def from_dict(self, dict_obj):
        """ Method that initialize from dict the TwoAxesChartTemplate class
        """
        # Deserialize annotation upper left attribute if exist
        if AbstractPostProcessingPlotlyTooling.ANNOTATION_UPPER_LEFT in dict_obj:
            self.annotation_upper_left = dict_obj[AbstractPostProcessingPlotlyTooling.ANNOTATION_UPPER_LEFT]

        # Deserialize annotation upper right attribute if exist
        if AbstractPostProcessingPlotlyTooling.ANNOTATION_UPPER_RIGHT in dict_obj:
            self.annotation_upper_right = dict_obj[AbstractPostProcessingPlotlyTooling.ANNOTATION_UPPER_RIGHT]

    def get_default_title_layout(self, title_name='', pos_x=0.5, pos_y=0.9):
        """ Generate plotly layout dict for title
            :params: title_name : title of chart
            :type: str
            :params: pos_x : position of title on x axis
            :type: float
            :params: pos_y : position of title on y axis
            :type: float

            :return: title_dict : dict that contains plotly layout for the title
            :type: dict
        """
        title_dict = {
            'text': f'<b>{title_name}</b>',
            'y': pos_y,
            'x': pos_x,
            'xanchor': 'center',
            'yanchor': 'top'}
        return title_dict

    def get_default_legend_layout(self, pos_x=0, pos_y=-0.2, orientation='h'):
        """ Generate plotly layout dict for legend
            :params: pos_x : position of legend on x axis
            :type: float
            :params: pos_y : position of legend on y axis
            :type: float
            :params: orientation : orientation of the legend
            :type: str

            :return: legend_dict : dict that contains plotly layout for the legend
            :type: dict
        """
        legend_dict = {'orientation': orientation,
                       'xanchor': 'left',
                       'yanchor': 'top',
                       'bgcolor': 'rgba(255, 255, 255, 0)',
                       'bordercolor': 'rgba(255, 255, 255, 0)',
                       'y': pos_y,
                       'x': pos_x}
        return legend_dict

    def get_default_font_layout(self):
        """ Generate plotly layout dict for font

            :return: font_dict : dict that contains plotly layout for the font
            :type: dict
        """
        font_dict = {'family': 'Arial',
                     'size': 10,
                     'color': '#7f7f7f'}
        return font_dict

    def get_default_annotations_upper_left_layout(self, pos_x=0, pos_y=1.15):
        """ Generate plotly layout list for upper left annotations
            :params: pos_x : position of upper left annotations on x axis
            :type: float
            :params: pos_y : position of upper left annotations on y axis
            :type: float

            :return: annotation_upper_left_dict : dict that contains plotly layout for the upper left annotations
            :type: dict
        """
        annotation_upper_left_dict = {}

        if len(self.annotation_upper_left) > 0:
            annotation_text_list = []

            for annotation_key in self.annotation_upper_left.keys():
                annotation_text_list.append(
                    f'{annotation_key}: {self.annotation_upper_left[annotation_key]}')

            annotation_upper_left_dict.update(
                {'text': '<br>'.join(annotation_text_list)})
            annotation_upper_left_dict.update({'align': 'left'})
            annotation_upper_left_dict.update({'showarrow': False})
            annotation_upper_left_dict.update({'xref': 'paper'})
            annotation_upper_left_dict.update({'yref': 'paper'})
            annotation_upper_left_dict.update({'x': pos_x})
            annotation_upper_left_dict.update({'y': pos_y})
            annotation_upper_left_dict.update({'bordercolor': 'black'})
            annotation_upper_left_dict.update({'borderwidth': 1})

        return annotation_upper_left_dict

    def get_default_annotations_upper_right_layout(self, pos_x=1, pos_y=1.15):
        """ Generate plotly layout list for upper right annotations
            :params: pos_x : position of upper right annotations on x axis
            :type: float
            :params: pos_y : position of upper right annotations on y axis
            :type: float

            :return: annotation_upper_right_dict : dict that contains plotly layout for the upper right annotations
            :type: dict
        """
        annotation_upper_right_dict = {}

        if len(self.annotation_upper_right) > 0:
            annotation_text_list = []

            for annotation_key in self.annotation_upper_right.keys():
                annotation_text_list.append(
                    f'{annotation_key}: {self.annotation_upper_right[annotation_key]}')

            annotation_upper_right_dict.update(
                {'text': '<br>'.join(annotation_text_list)})
            annotation_upper_right_dict.update({'align': 'right'})
            annotation_upper_right_dict.update({'showarrow': False})
            annotation_upper_right_dict.update({'xref': 'paper'})
            annotation_upper_right_dict.update({'yref': 'paper'})
            annotation_upper_right_dict.update({'x': pos_x})
            annotation_upper_right_dict.update({'y': pos_y})
            annotation_upper_right_dict.update({'bordercolor': 'black'})
            annotation_upper_right_dict.update({'borderwidth': 1})

        return annotation_upper_right_dict

    def set_csv_data(self, plot_csv_data):
        """ Set value of class variable plot_csv_data
            :params: plot_csv_data : all csv lines of plot data
            :type: list of str
        """
        if plot_csv_data is not None:
            if isinstance(plot_csv_data, list):
                if all(isinstance(s, str) for s in plot_csv_data):
                    self._plot_csv_data = plot_csv_data
                else:
                    raise ValueError(
                        'Parameter plot_csv_data elements must all be str')
            else:
                raise ValueError(
                    f'Parameter plot_csv_data is not a list => type({type(plot_csv_data)})')
        else:
            raise ValueError('Parameter plot_csv_data cannot be None')

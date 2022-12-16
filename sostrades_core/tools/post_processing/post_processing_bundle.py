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
Post processing bundle model
"""


class PostProcessingBundle:
    """ Class that hold filter and post processing bundle
    """

    NAME = 'name'
    DISCIPLINE_NAME = 'discipline_name'
    FILTERS = 'filters'
    POST_PROCESSINGS = 'post_processings'

    def __init__(self, name, discipline_name, filters, post_processings):
        """ Constructor

        :params: name, name of current post processings bundle
        :type: str
        :params: discipline_name, name of current discipline hosting the post processings bundle
        :type: str
        :params: filters, filter list used for this post-processings bundle
        :type: ChartFilter[]
        :params: post_processings, list of post-processings bundle
        :type: post processing*[]

        """

        self.name = name
        self.discipline_name = discipline_name
        self.filters = filters
        self.post_processings = post_processings

    @property
    def has_post_processings(self):
        return len(self.filters) > 0

    def __repr__(self):
        """ Overload of the class representation
        """

        series_string = [f'\nname: {self.name}',
                         f'discipline_name: {self.discipline_name}',
                         f'filters: {self.filters}',
                         f'post-processings: {self.post_processings}'
                         ]

        return '\n'.join(series_string)

    def to_dict(self):
        """ Method that serialize as dict the SeriesTemplate class
        """

        dict_obj = {}
        # Serialize name attribute
        dict_obj.update({PostProcessingBundle.NAME: self.name})

        # Serialize discipline name attribute
        dict_obj.update({PostProcessingBundle.DISCIPLINE_NAME: self.discipline_name})

        # Serialize filters parameter attribute
        dict_obj.update(
            {PostProcessingBundle.FILTERS: self.filters})

        # Serialize post-processings values parameter attribute
        dict_obj.update(
            {PostProcessingBundle.POST_PROCESSINGS: self.post_processings})

        return dict_obj

    @staticmethod
    def from_dict(dict_obj):
        """ Method that initialize from dict the SeriesTemplate class
        """

        # Serialize name attribute
        name = dict_obj[PostProcessingBundle.NAME]

        # Deserialize discipline name attribute
        discipline_name = ''
        if PostProcessingBundle.DISCIPLINE_NAME in dict_obj:
            discipline_name = dict_obj[PostProcessingBundle.DISCIPLINE_NAME]


        # Serialize filters parameter attribute
        filters = dict_obj[PostProcessingBundle.FILTERS]

        # Serialize post-processings values parameter attribute
        post_processings = dict_obj[PostProcessingBundle.POST_PROCESSINGS]

        return PostProcessingBundle(name, discipline_name, filters, post_processings)

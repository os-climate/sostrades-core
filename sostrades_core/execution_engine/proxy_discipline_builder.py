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
from copy import deepcopy

from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''

from abc import abstractmethod


class ProxyDisciplineBuilderException(Exception):
    pass


class ProxyDisciplineBuilder(ProxyDiscipline):
    """**ProxyDisciplineBuilder** is a ProxyDiscipline with the ability to instantiate sub proxies. It is an
    abstract clas that delegates the build method to the children classes.

    All nodes of the SoSTrades process tree that have sub proxies are represented by proxy instances that inherit
    from ProxyDisciplineBuilder (e.g. ProxyCoupling).

    Attributes:
        proxy_discipline(List[ProxyDiscipline]): list of sub proxies managed by the discipline (case  a proxyCoupling)
        built_sos_disciplines(List[ProxyDiscipline]): list of sub proxies instanciated by the discilpline (case of
        scatter, archibuilder,...)

    """

    # -- Disciplinary attributes

    # ontology information
    _ontology_data = {
        'label': 'Core Discipline Builder Model',
        'type': 'Official',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': '',
        'version': '',
    }

    # def __init__(self):
    #
    #     self.proxy_discipline = None
    #     self.built_sos_disciplines = None

    @abstractmethod
    def build(self):
        """
        To be overloaded by subclasses
        Builds sub processes (i.e., in case of scatters, ...)"""

        # self.proxy_discipline.build()

    def clean(self):
        """
        This method cleans a sos_discipline_builder, which is a discipline that can build other disciplines;
        We first begin by cleaning all the disciplines children, afterward we clean the discipline itself
        """
        for discipline in self.built_sos_disciplines:
            discipline.clean()
            self.ee.factory.remove_discipline_from_father_executor(discipline)

        #         SoSDiscipline.clean(self)
        self.father_builder.remove_discipline(self)
        self.clean_dm_from_disc()
        self.ee.ns_manager.remove_dependencies_after_disc_deletion(
            self, self.disc_id)
        self.ee.factory.remove_sos_discipline(self)

    def clean_children(self, list_children):
        """
        This method cleans the given list of children from the current discipline
        """
        for discipline in list_children:
            self.built_sos_disciplines.remove(discipline)
            discipline.clean()
            self.ee.factory.remove_discipline_from_father_executor(discipline)

    def get_desc_in_out(self, io_type):
        """
        Retrieves information from wrapper or ProxyDiscipline DESC_IN to fill data_in
        Overload of proxyDiscipline

        Argument:
                io_type : 'string' . indicates whether we are interested in desc_in or desc_out
        """
        if io_type == self.IO_TYPE_IN:
            return deepcopy(self.DESC_IN) or {}
        elif io_type == self.IO_TYPE_OUT:
            return deepcopy(self.DESC_OUT) or {}
        else:
            raise Exception(
                f'data type {io_type} not recognized [{self.IO_TYPE_IN}/{self.IO_TYPE_OUT}]')

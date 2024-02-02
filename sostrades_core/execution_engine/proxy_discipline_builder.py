'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/26-2023/11/03 Copyright 2023 Capgemini

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
    PROPAGATE_CACHE = 'propagate_cache_to_children'
    NUM_DESC_IN = {PROPAGATE_CACHE: {ProxyDiscipline.TYPE: 'bool', ProxyDiscipline.POSSIBLE_VALUES: [True, False],
                                     ProxyDiscipline.NUMERICAL: True,
                                     ProxyDiscipline.STRUCTURING: True},
                   }
    NUM_DESC_IN.update(ProxyDiscipline.NUM_DESC_IN)

    def _build(self, builder_list):
        """
        Instanciate sub proxies managed by the coupling
        """
        old_current_discipline = self.ee.factory.current_discipline
        self.set_father_discipline()

        for builder in builder_list:
            # A builder of disciplines should propagate its associated namespaces
            # and has the priority over associated namespaces already set
            self.associate_namespace_to_sub_builder(builder)

            proxy_disc = builder.build()

            if self.ee.ns_manager.get_local_namespace(
                    self).is_display_value() and builder not in self.ee.ns_manager.display_ns_dict:
                father_display_value = self.get_disc_display_name()
                display_value = f'{father_display_value}.{builder.sos_name}'
                self.ee.ns_manager.get_local_namespace(
                    proxy_disc).set_display_value(display_value)

            if proxy_disc not in self.proxy_disciplines:
                self.ee.factory.add_discipline(proxy_disc)
        # If the old_current_discipline is None that means that it is the first build of a coupling then self is the
        # high level coupling and we do not have to restore the
        # current_discipline
        if old_current_discipline is not None:
            self.ee.factory.current_discipline = old_current_discipline

    #     def clear_cache(self):
    #         self.mdo_chain.cache.clear()
    #         ProxyDisciplineBuilder.clear_cache(self)

    def associate_namespace_to_sub_builder(self, builder):
        '''
        Associate namespaces in associated namespaces list to sub builders
        '''
        if self.associated_namespaces != []:
            builder.add_namespace_list_in_associated_namespaces(
                self.associated_namespaces)

    def set_father_discipline(self):
        '''
        Set the current discipline to build the builder_list at this level
        '''
        self.ee.factory.current_discipline = self

    def build(self):
        builder_list = self.prepare_build()
        self._build(builder_list)

    def prepare_build(self):
        """
        To be overload by subclasses with special builds.
        """
        return self.cls_builder

    def update_data_io_with_child(self, sub_data_in, sub_data_out):
        '''

        Args:
            sub_data_in: data_in of the child under the builder
            sub_data_out: data_out of the child under the builder

        Returns:
            Update the _data_io and the _simple_data_io of the proxydisciplinecbuilder accridng to its children

        '''
        if sub_data_in != {}:
            self._update_data_io(sub_data_in, self.IO_TYPE_IN)
            self.build_simple_data_io(self.IO_TYPE_IN)
        if sub_data_out != {}:
            self._update_data_io(sub_data_out, self.IO_TYPE_OUT)
            self.build_simple_data_io(self.IO_TYPE_OUT)

    def clean(self):
        """
        This method cleans a sos_discipline_builder, which is a discipline that can build other disciplines;
        We first begin by cleaning all the disciplines children, afterward we clean the discipline itself
        """
        self.clean_children()

        super().clean()

    def clean_children(self, list_children=None):
        """
        This method cleans the given list of children from the current discipline
        If no given_list then it clean the entire list of built_proxy_disciplines
        """
        if list_children is not None:
            for discipline in list_children:
                discipline.clean()
                self.ee.factory.remove_discipline_from_father_executor(
                    discipline)
        else:
            for discipline in self.proxy_disciplines:
                discipline.clean()

            self.proxy_disciplines = []

        self._is_configured = False

    def create_sub_builder_coupling(self, builder_name, sub_builders, father_display_value=None):

        disc_builder = self.ee.factory.create_builder_coupling(builder_name)

        disc_builder.set_builder_info('cls_builder', sub_builders)

        if father_display_value is None:
            father_display_value = self.get_disc_display_name()

        display_value = f'{father_display_value}.{disc_builder.sos_name}'

        self.ee.ns_manager.add_display_ns_to_builder(
            disc_builder, display_value)

        return disc_builder

    def set_children_numerical_inputs(self):
        """
        Set numerical inputs values (cache_type, cache_file_path, debug_mode, linearization_mode) for those who have
        changed
        """
        if self.PROPAGATE_CACHE in self.get_data_in():
            propagate_cache_to_children = self.get_sosdisc_inputs(
                self.PROPAGATE_CACHE)
            if self._reset_cache and self._set_children_cache and propagate_cache_to_children:
                cache_type = self.get_sosdisc_inputs(
                    ProxyDiscipline.CACHE_TYPE)
                cache_file_path = self.get_sosdisc_inputs(
                    ProxyDiscipline.CACHE_FILE_PATH)
                for disc in self.proxy_disciplines:
                    disc_in = disc.get_data_in()
                    if ProxyDiscipline.CACHE_TYPE in disc_in:
                        self.dm.set_data(disc.get_var_full_name(
                            ProxyDiscipline.CACHE_TYPE, disc_in), self.VALUE, cache_type, check_value=False)
                        if cache_file_path is not None:
                            self.dm.set_data(disc.get_var_full_name(
                                ProxyDiscipline.CACHE_FILE_PATH, disc_in), self.VALUE, cache_file_path,
                                check_value=False)
                    if self.PROPAGATE_CACHE in disc_in:
                        self.dm.set_data(disc.get_var_full_name(
                            self.PROPAGATE_CACHE, disc_in), self.VALUE, propagate_cache_to_children, check_value=False)
                self._set_children_cache = False

            if self._reset_debug_mode:
                self.set_debug_mode_rec(
                    self.get_sosdisc_inputs(ProxyDiscipline.DEBUG_MODE))
                self._reset_debug_mode = False

            if self._reset_linearization_mode:
                self.set_linearization_mode_rec(
                    linearization_mode=self.get_sosdisc_inputs(ProxyDiscipline.LINEARIZATION_MODE))
                self._reset_linearization_mode = False

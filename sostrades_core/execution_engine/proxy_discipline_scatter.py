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
'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''
import numpy as np

from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.execution_engine.proxy_discipline_builder import ProxyDisciplineBuilder


class SoSDisciplineScatterException(Exception):
    pass


class ProxyDisciplineScatter(ProxyDisciplineBuilder):
    '''
    Class that build disciplines using a builder and a map containing data to scatter
    '''

    # ontology information
    _ontology_data = {
        'label': 'Scatter',
        'type': 'Official',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-indent fa-fw',
        'version': '',
    }

    def __init__(self, sos_name, ee, map_name, cls_builder, associated_namespaces=None, coupling_per_scatter=False):
        '''
        Constructor
        '''
        self.__factory = ee.factory
        self.__scattered_disciplines = {}
        self.sub_coupling_builder_dict = {}
        self.__gathered_disciplines = {}
        self.__scatter_data_map = []
        self.__scatter_build_map = []

        self._maturity = ''

        self.coupling_per_scatter = coupling_per_scatter
        # associate map to discipline
        self.map_name = map_name
        self.sc_map = ee.smaps_manager.get_build_map(self.map_name)
        ee.smaps_manager.associate_disc_to_build_map(self)
        self.sc_map.configure_map(cls_builder)
        self.__builders = cls_builder
        # if isinstance(self.__builders)
        ProxyDisciplineBuilder.__init__(
            self, sos_name, ee, associated_namespaces=associated_namespaces)
        # add input_name to inst_desc_in
        self.build_inst_desc_in_with_map()
        self.builder_name = None
        if not isinstance(self.__builders, list):
            self.builder_name = self.__builders.sos_name

    def get_scattered_disciplines(self):
        return self.__scattered_disciplines

    @property
    def scatter_data_map(self):
        return self.__scatter_data_map

    @property
    def scatter_builders(self):
        return self.__builders

    @property
    def scatter_build_map(self):
        return self.__scatter_build_map

    def build_inst_desc_in_with_map(self):
        '''
        Consult the associated scatter map and adapt the inst_desc_in of the gather with the scatter var_name 
        '''
        input_name = self.sc_map.get_input_name()
        input_type = 'list'
        input_subtype_descriptor = {'list': 'string'}

        if self.sc_map.INPUT_NS in self.sc_map.get_map():
            scatter_desc_in = {input_name: {
                ProxyDiscipline.TYPE: input_type, ProxyDiscipline.SUBTYPE: input_subtype_descriptor,
                ProxyDiscipline.VISIBILITY: ProxyDiscipline.SHARED_VISIBILITY,
                ProxyDisciplineBuilder.NAMESPACE: self.sc_map.get_input_ns(), ProxyDiscipline.STRUCTURING: True,
                ProxyDiscipline.USER_LEVEL: 3}}
        else:
            scatter_desc_in = {input_name: {
                ProxyDiscipline.TYPE: input_type, ProxyDiscipline.SUBTYPE: input_subtype_descriptor,
                ProxyDiscipline.VISIBILITY: ProxyDiscipline.LOCAL_VISIBILITY,
                ProxyDiscipline.STRUCTURING: True,
                ProxyDiscipline.USER_LEVEL: 3}}

        self.inst_desc_in.update(scatter_desc_in)

    def build(self):
        ''' 
        Configuration of the SoSscatter : 
        -First configure the scatter 
        -Get the list to scatter on and the associated namespace
        - Look if disciplines are already scatterred and compute the new list to scatter (only new ones)
        - Remove disciplines that are not in the scatter list
        - Scatter the instantiator cls and adapt namespaces depending if it is a list or a singleton
        '''

        # old_current_discipline = self.ee.factory.current_discipline
        # self.ee.factory.current_discipline = self

        self.ee.ns_manager.update_shared_ns_with_others_ns(self)
        input_name = self.sc_map.get_input_name()  # ac_name_list
        local_namespace = self.ee.ns_manager.get_local_namespace_value(
            self)

        ns_to_update = self.sc_map.get_ns_to_update()
        # store ns_to_update namespace object
        old_ns_to_update = {}
        for ns_name in ns_to_update:
            old_ns_to_update[ns_name] = self.ee.ns_manager.get_shared_namespace(self,
                                                                                ns_name)

        # remove all disciplines not in subproc_names
        if input_name in self.get_data_in().keys():
            sub_names = self.get_sosdisc_inputs(
                input_name)  # [ac1, ac2, ...]
            if sub_names is not None:

                new_sub_names = self.clean_scattered_disciplines(sub_names)

                # build sub_process through the factory
                for name in sub_names:
                    if self.coupling_per_scatter:
                        self.build_sub_coupling(
                            name, local_namespace, new_sub_names, old_ns_to_update)
                    else:
                        self.build_child_scatter(
                            name, local_namespace, new_sub_names, old_ns_to_update)

                self.ee.ns_manager.shared_ns_dict.update(old_ns_to_update)

        # if old_current_discipline is not None:
        #     self.ee.factory.current_discipline = old_current_discipline

    def build_sub_coupling(self, name, local_namespace, new_sub_names, old_ns_to_update):

        # Call scatter map to modify the associated namespace
        self.sc_map.modify_scatter_ns(self.builder_name, name, local_namespace)

        self.sc_map.update_ns(
            old_ns_to_update, name, self.sos_name)

        if name in new_sub_names:

            coupling_builder = self.ee.factory.create_builder_coupling(name)

            coupling_builder.set_builder_info('cls_builder', self.__builders)
            self.sub_coupling_builder_dict[name] = coupling_builder

            coupling_disc = coupling_builder.build()
            # flag the coupling so that it can be executed in parallel
            coupling_disc.is_parallel = True
            self.add_scatter_discipline(coupling_disc, name)

        else:
            coupling_disc = self.sub_coupling_builder_dict[name].build()
            # flag the coupling so that it can be executed in parallel
            coupling_disc.is_parallel = True

    def build_child_scatter(self, name, local_namespace, new_sub_names, old_ns_to_update):

        # Call scatter map to modify the associated namespace
        ns_ids = []

        ns_scatter_id = self.sc_map.modify_scatter_ns(
            self.builder_name, name, local_namespace)
        ns_ids.append(ns_scatter_id)
        ns_update_ids = self.sc_map.update_ns(
            old_ns_to_update, name, self.sos_name)
        ns_ids.extend(ns_update_ids)
        # Case of a scatter of coupling :
        if isinstance(self.__builders, list):
            self.build_scatter_of_coupling(
                name, local_namespace, new_sub_names, ns_update_ids)

        # Case of a coupling of scatter :
        else:
            self.build_coupling_of_scatter(name, new_sub_names, ns_update_ids)

    def build_scatter_of_coupling(self, name, local_namespace, new_sub_names, ns_update_ids):
        '''
        # We set the scatter_name in the namespace and the discipline is called with its origin name
        #
        # scatter
        #        |_name_1
        #                |_Disc1
        #                |_Disc2
        #        |_name_2
        #                |_Disc1
        #                |_Disc2

        '''
        for builder in self.__builders:
            self.ee.ns_manager.set_current_disc_ns(
                f'{local_namespace}.{name}')
            if builder.associated_namespaces != []:
                builder.add_namespace_list_in_associated_namespaces(
                    self.associated_namespaces)

            builder.add_namespace_list_in_associated_namespaces(
                ns_update_ids)
            disc = builder.build()
            # Add the discipline only if it is in
            # new_sub_names
            if name in new_sub_names:
                self.add_scatter_discipline(disc, name)

    def build_coupling_of_scatter(self, name, new_sub_names, ns_update_ids):
        '''
        # We set the scatter_name as the discipline name in the scatter which has already the name of the builder
        # Disc1 is the scatter
        #
        # Disc1
        #        |_name_1
        #        |_name_2
        '''
        old_builder_name = self.__builders.sos_name
        self.__builders.set_disc_name(name)
        if self.__builders.associated_namespaces != []:
            self.__builders.add_namespace_list_in_associated_namespaces(
                self.associated_namespaces)

        self.__builders.add_namespace_list_in_associated_namespaces(
            ns_update_ids)
        disc = self.__builders.build()
        self.__builders.set_disc_name(old_builder_name)

        # Add the discipline only if it is in
        # new_sub_names
        if name in new_sub_names:
            self.add_scatter_discipline(disc, name)

    def clean_scattered_disciplines(self, sub_names):
        '''
        Clean disciplines that was scattered and are not in the scatter_list anymore
        Return the new scatter names not yet present in the list of scattered disciplines
        '''
        # sort sub_names to filter new names and disciplines to remove
        new_sub_names = [
            name for name in sub_names if not name in self.__scattered_disciplines]
        disc_name_to_remove = [
            name for name in self.__scattered_disciplines if not name in sub_names]
        for disc_name in disc_name_to_remove:
            self.clean_children(self.__scattered_disciplines[disc_name])
            if self.coupling_per_scatter:
                del self.sub_coupling_builder_dict[disc_name]

            del self.__scattered_disciplines[disc_name]

        return new_sub_names

    def add_disc_to_remove_recursive(self, disc, to_remove):
        if isinstance(disc, ProxyDisciplineScatter):
            to_remove.append(disc)
            for disc_list in disc.__scattered_disciplines.values():
                for sub_disc in disc_list:
                    self.add_disc_to_remove_recursive(sub_disc, to_remove)
        elif type(disc).__name__ == 'SoSMultiScatterBuilder':
            to_remove.append(disc)
            for builder in np.concatenate(list(disc.child_scatter_builder_list.values())):
                for sub_disc in list(builder.discipline_dict.values()):
                    self.add_disc_to_remove_recursive(
                        sub_disc, to_remove)
        else:
            to_remove.append(disc)

    def remove_scattered_disciplines(self, disc_to_remove):
        '''
        Remove a list of disciplines from the scattered_disciplines
        '''

        for disc in disc_to_remove:
            if self.coupling_per_scatter:
                del self.sub_coupling_builder_dict[disc]

            del self.__scattered_disciplines[disc]

    def add_scatter_discipline(self, disc, name):
        '''
        Add the discipline to the factory and to the dictionary of scattered_disciplines
        '''
        self.__factory.add_discipline(disc)
        if name in self.__scattered_disciplines.keys():
            self.__scattered_disciplines[name].append(disc)
        else:
            self.__scattered_disciplines.update({name: [disc]})
        if disc not in self.built_proxy_disciplines:
            self.built_proxy_disciplines.append(disc)

    def get_maturity(self):
        '''
        Get the maturity of the scatter TODO
        '''
        return ''

    def run(self):
        '''
        No run function for a SoSDisciplineScatter
        '''
        pass

    def setup_sos_disciplines(self):
        """
        Method to be overloaded to add dynamic inputs/outputs using add_inputs/add_outputs methods.
        If the value of an input X determines dynamic inputs/outputs generation, then the input X is structuring and the item 'structuring':True is needed in the DESC_IN
        DESC_IN = {'X': {'structuring':True}}
        """
        pass

    def prepare_execution(self):
        """
        Overload
        """
        self._status = ProxyDiscipline.STATUS_PENDING

    def get_status_after_configure(self):
        if len(self.built_proxy_disciplines) > 0:
            if any(proxy.status == ProxyDiscipline.STATUS_FAILED for proxy in self.built_proxy_disciplines):
                return ProxyDiscipline.STATUS_FAILED
            return self.built_proxy_disciplines[-1].status
        return ProxyDiscipline.STATUS_DONE

    def create_mdo_discipline_wrap(self, name, wrapper, wrapping_mode):
        """
        overload of proxy_discipline's method
        """
        pass

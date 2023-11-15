'''
Copyright 2022 Airbus SAS
Modifications on 2023/07/17-2023/11/03 Copyright 2023 Capgemini

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
from sostrades_core.execution_engine.builder_tools.sos_tool import SosTool
from collections import defaultdict

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''


class ScatterTool(SosTool):
    '''
    Class that build disciplines using a builder and a map containing data to scatter
    '''

    # ontology information
    _ontology_data = {
        'label': 'Scatter Tool',
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
    DISPLAY_OPTIONS_POSSIBILITIES = ['hide_under_coupling', 'hide_coupling_in_driver',
                                     'group_scenarios_under_disciplines']

    #            display_options (optional): Dictionary of display_options for multiinstance mode (value True or False) with options :
    #             'hide_under_coupling' : Hide all disciplines created under the coupling at scenario name node for display purpose
    #             'hide_coupling_in_driver': Hide the coupling (scenario_name node) under the driver for display purpose
    #             'group_scenarios_under_disciplines' : Invert the order of scenario and disciplines for display purpose
    #                                                   Scenarios will be under discipline for the display treeview

    def __init__(self, sos_name, ee, cls_builder, map_name=None):
        '''
        Constructor
        '''

        SosTool.__init__(self, sos_name, ee, cls_builder)

        self.map_name = map_name
        self.driver_display_value = None
        self.__scattered_disciplines = {}
        self.__gather_disciplines = {}
        self.__scatter_list = None
        self.input_name = None
        self.ns_to_update = None
        self.sc_map = None
        self.display_options = {disp_option: False for disp_option in self.DISPLAY_OPTIONS_POSSIBILITIES}

    @property
    def has_built(self):
        return self.__scattered_disciplines.keys() == set(self.__scatter_list)

    @property
    def scatter_list(self):
        return self.__scatter_list

    def set_display_options(self, display_options_dict):
        '''
        Set the display options dictionnary for the driver
        '''

        if display_options_dict is not None:
            if not isinstance(display_options_dict, dict):
                raise Exception(
                    'The display options parameter for the driver creation should be a dict')
            else:
                for key in display_options_dict:
                    if key not in self.DISPLAY_OPTIONS_POSSIBILITIES:
                        raise Exception(
                            f'Display options should be in the possible list : {self.DISPLAY_OPTIONS_POSSIBILITIES}')
                    else:
                        self.display_options[key] = display_options_dict[key]

    def associate_tool_to_driver(self, driver, cls_builder=None, associated_namespaces=None):
        '''    
        Method that associate tool to the driver and add scatter map
        '''
        SosTool.associate_tool_to_driver(
            self, driver, cls_builder=cls_builder, associated_namespaces=associated_namespaces)

        if self.map_name is not None:
            self.sc_map = self.ee.scattermap_manager.get_build_map(self.map_name)
            self.ee.scattermap_manager.associate_disc_to_build_map(self)
            self.sc_map.configure_map(self.sub_builders)
        # get initial values of namespaces before updat eby the scatter tool at each build
        self.get_values_for_namespaces_to_update()

    def prepare_tool(self):
        '''
        Prepare tool function if some data of the driver are needed to configure the tool
        '''

        super().prepare_tool()
        if self.driver.SAMPLES_DF in self.driver.get_data_in():
            instance_reference = self.driver.get_sosdisc_inputs(self.driver.INSTANCE_REFERENCE)
            samples_df = self.driver.get_sosdisc_inputs(self.driver.SAMPLES_DF)
            # sce_df = copy.deepcopy(samples_df)
            if instance_reference:
                samples_df = samples_df.append(
                    {self.driver.SELECTED_SCENARIO: True, self.driver.SCENARIO_NAME: 'ReferenceScenario'},
                    ignore_index=True)

            self.set_scatter_list(
                samples_df[samples_df[self.driver.SELECTED_SCENARIO] == True][
                    self.driver.SCENARIO_NAME].values.tolist())

        display_options = self.driver.get_sosdisc_inputs('display_options')
        # if display options are set in the process, it wins we cannot modify display options again
        if self.driver.display_options is not None:
            self.set_display_options(self.driver.display_options)
        # else we check if the input has changed
        else:
            if display_options is not None:
                self.set_display_options(display_options)

        if self.display_options['hide_coupling_in_driver']:
            self.driver_display_value = self.driver.get_disc_display_name()

    def get_values_for_namespaces_to_update(self):
        '''
        Get the values of the namespace list defined in the namespace manager
        '''
        ns_to_update_name_list = self.get_ns_to_update_name_list()

        # store ns_to_update namespace object
        self.ns_to_update = {}

        for ns_name in ns_to_update_name_list:
            # we should take ns_to_update of the shared_ns_dict to be consistent with father_executor name and driver_name
            self.ns_to_update[ns_name] = self.ee.ns_manager.get_ns_in_shared_ns_dict(ns_name)

    def get_dynamic_output_from_tool(self):
        '''
        Add the scatter list output name into dynamic desc_out in the behalf of the driver
        this scatter_list is depending on samples_df configuration
        Add then all scenario_name for each scenario
        '''
        dynamic_outputs = {}
        if self.sc_map is not None and self.sc_map.get_scatter_list_name_and_namespace() is not None:
            scatter_list_name, scatter_list_ns = self.sc_map.get_scatter_list_name_and_namespace()
            dynamic_outputs = {scatter_list_name: {'type': 'list',
                                                   'visibility': 'Shared',
                                                   'namespace': scatter_list_ns,
                                                   'value': self.__scatter_list}}
        if self.sc_map is not None and self.sc_map.get_scatter_name() is not None:
            scatter_name = self.sc_map.get_scatter_name()
            for scatter_value in self.__scatter_list:
                dynamic_outputs.update({f'{scatter_value}.{scatter_name}':
                                            {'type': 'string',
                                             'value': scatter_value}})
        return dynamic_outputs

    def get_ns_to_update_name_list(self):
        '''
        Returns the list of namespace to update depending if there is a scatter map or not
        If there is a scatter map build the list with ns_to_update or ns_not_to_update
        If not build the list with namespaces in sub builders
        '''
        if self.sc_map is None:
            ns_to_update_name_list = self.driver.sub_builder_namespaces
        else:
            if self.sc_map.is_ns_to_update_or_not() is None:
                ns_to_update_name_list = self.driver.sub_builder_namespaces
            elif self.sc_map.is_ns_to_update_or_not():
                ns_to_update_name_list = self.sc_map.get_ns_to_update()
            else:
                ns_not_to_update_name_list = self.sc_map.get_ns_not_to_update()
                ns_to_update_name_list = [ns_name for ns_name in self.driver.sub_builder_namespaces if
                                          ns_name not in ns_not_to_update_name_list]
        return ns_to_update_name_list

    def set_scatter_list(self, scatter_list):
        self.__scatter_list = scatter_list
        if self.sc_map is not None and self.sc_map.get_scatter_list_name_and_namespace() is not None:
            scatter_list_name, scatter_list_namespace = self.sc_map.get_scatter_list_name_and_namespace()
            if scatter_list_name in self.driver.get_data_out():
                self.ee.dm.set_data(self.driver.get_var_full_name(
                    scatter_list_name, self.driver.get_data_out()), 'value', self.__scatter_list)

    def build(self):
        ''' 
        Configuration of the SoSscatter : 
        -First configure the scatter 
        -Get the list to scatter on and the associated namespace
        - Look if disciplines are already scatterred and compute the new list to scatter (only new ones)
        - Remove disciplines that are not in the scatter list
        - Scatter the instantiator cls and adapt namespaces depending if it is a list or a singleton
        '''

        if self.__scatter_list is not None:

            # get new_names that are not yet built and clean the one that are no more in the scatter list
            new_sub_names = self.clean_scattered_disciplines(
                self.__scatter_list)

            if len(self.__scatter_list) > 0:
                # Always add gather to get gather_outputs
                self.add_gather()

            for name in self.__scatter_list:
                # check if the name is new
                new_name_flag = name in new_sub_names

                self.build_child(
                    name, new_name_flag)

    def update_namespaces(self, name):
        '''
        Update all ns_to_update namespaces and the scatter namespace with the scatter_name just after the local_namespace
        Return the list of namespace keys for future builder association
        All namespaces are not added in shared_ns_dict to be transparent and only associated to the right disciplines
        We need to take into account all namespaces already associated to builders that needs a specific update by builders
        '''

        ns_ids_list = []
        extra_name = f'{self.driver.sos_name}.{name}'
        after_name = self.driver.father_executor.get_disc_full_name()

        for ns_name, ns in self.ns_to_update.items():
            updated_value = self.ee.ns_manager.update_ns_value_with_extra_ns(
                ns.get_value(), extra_name, after_name=after_name)
            display_value = ns.get_display_value_if_exists()
            ns_id = self.ee.ns_manager.add_ns(
                ns_name, updated_value, display_value=display_value, add_in_shared_ns_dict=False)
            ns_ids_list.append(ns_id)

        return ns_ids_list

    def build_child(self, name, new_name_flag):
        '''
        #        |_name_1
        #                |_Disc1
        #                |_Disc2
        #        |_name_2
        #                |_Disc1
        #                |_Disc2

        Build child disciplines under the father executor of the driver (to get a flatten subprocess all the time)
        name (string) : new name in the scatter_list
        new_name_flag (bool) : True if name is a new_name in the build
        ns_ids_list (list) : The list of ns_keys that already have been updated with the scatter_name and mus tbe associated to the builder 

        1. Set builders as a list and loop over builders
        2. Set the new_name of the builder with the scatter name
        3. Associate new namespaces to the builder (the coupling will associate the namespace to its children)
        4. Build the builder
        5 Add the builded discipline to the driver and factory
        6. Set the old name to the builder for next iteration

        '''

        for builder in self.sub_builders:

            old_builder_name = builder.sos_name
            disc_name = self.get_subdisc_name(name, old_builder_name)

            builder.set_disc_name(disc_name)

            if new_name_flag:
                # update namespaces to update with this name
                ns_ids_list = self.update_namespaces(name)
                self.associate_namespaces_to_builder(builder, ns_ids_list)
            self.set_father_discipline()
            disc = builder.build()

            self.apply_display_options(disc, name, old_builder_name)

            builder.set_disc_name(old_builder_name)
            # Add the discipline only if it is a new_name
            if new_name_flag:
                self.add_scatter_discipline(disc, name)

    def get_subdisc_name(self, name, old_builder_name):
        '''

        Args:
            name: name of the scenario
            old_builder_name: old name of the builder

        Returns:
            disc_name : full_name of the discipline to build

        '''
        # get the full_name of the driver and of the father_executor
        driver_full_name = self.driver.get_disc_full_name()
        father_executor_name = self.driver.father_executor.get_disc_full_name()
        # delete the name of the father_executor because the disc will be built at father_executor node
        namespace_name = driver_full_name.replace(f'{father_executor_name}.', '', 1)
        disc_name = f'{namespace_name}.{name}.{old_builder_name}'

        return disc_name

    def apply_display_options(self, disc, name, old_builder_name):
        '''
        Apply the display options proposed by the driver in multiinstance mode
        1. hide_under_coupling  : Hide all the disicplines instanciated under the high level coupling
        2. group_scenarios_under_disciplines : Group All the scenario under each discipline in display treeview (the exec treeview remains the same)
        3. autogather : Add a Gather discipline which will autogather disciplines
        '''
        driver_display_name = self.driver.get_disc_display_name()
        driver_full_name = self.driver.get_disc_full_name()
        if driver_display_name != driver_full_name:
            local_ns_disc = self.ee.ns_manager.get_local_namespace(disc)
            display_value = f'{driver_display_name}.{name}.{old_builder_name}'
            local_ns_disc.set_display_value(display_value)

        if self.display_options['hide_under_coupling']:
            local_ns_disc = self.ee.ns_manager.get_local_namespace(disc)
            display_value = f'{driver_display_name}.{name}'
            local_ns_disc.set_display_value(display_value)
        elif self.display_options['group_scenarios_under_disciplines']:
            local_ns_disc = self.ee.ns_manager.get_local_namespace(disc)
            display_value = f'{driver_display_name}.{old_builder_name}.{name}'
            local_ns_disc.set_display_value(display_value)
        elif self.display_options['hide_coupling_in_driver']:
            local_ns_disc = self.ee.ns_manager.get_local_namespace(disc)
            display_value = f'{driver_display_name}.{old_builder_name}'
            local_ns_disc.set_display_value(display_value)

        # if self.display_options['group_scenarios_under_disciplines']:
        #     gather_name = old_builder_name
        # else:
        gather_name = f'{self.driver.sos_name}_gather'

        self.__gather_disciplines[gather_name].add_disc_to_config_dependency_disciplines(disc)

    def add_gather(self):
        '''
            Add gather discipline for gather outputs
            the gather discipline name will automatically be the name of the builder
            if display option group_scenarios_under_disciplines is activated then we want a gather per subbuilder

        '''

        # if self.display_options['group_scenarios_under_disciplines']:
        #     for sub_builder in self.sub_builders:

        #         gather_name = f'{self.driver.sos_name}.{sub_builder.sos_name}'
        #         if sub_builder.sos_name not in self.__gather_disciplines:
        #             gather_builder = self.ee.factory.add_gather_builder(gather_name)

        #             # deal with ns_gather namespace
        #             if ns_driver is not None:
        #                 ns_gather = self.ee.ns_manager.add_ns('ns_gather', ns_driver)
        #                 gather_builder.associate_namespaces(ns_gather)

        #             self.set_father_discipline()
        #             gather_disc = gather_builder.build()
        #             self.ee.factory.add_discipline(gather_disc)
        #             self.__gather_disciplines[sub_builder.sos_name] = gather_disc
        # else:
        gather_name = f'{self.driver.sos_name}_gather'
        gather_path = f'{self.driver.get_disc_full_name()}_gather'
        # strip_first_ns
        gather_path = gather_path.split('.', 1)[1]
        if gather_name not in self.__gather_disciplines:
            gather_builder = self.ee.factory.add_gather_builder(gather_path)
            self.ee.ns_manager.add_display_ns_to_builder(
                gather_builder, self.driver.get_disc_display_name())

            self.set_father_discipline()
            gather_disc = gather_builder.build()
            self.ee.factory.add_discipline(gather_disc)
            self.__gather_disciplines[gather_name] = gather_disc

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
        self.remove_scattered_disciplines(disc_name_to_remove)

        if len(disc_name_to_remove) != 0 or len(new_sub_names) != 0:
            gather_discs = self.get_all_gather_disciplines()
            if len(gather_discs) != 0:
                for gather in gather_discs:
                    gather.set_configure_status(False)
        return new_sub_names

    def remove_scattered_disciplines(self, disc_to_remove):
        '''
        Remove a list of disciplines from the scattered_disciplines
        '''

        for disc in disc_to_remove:
            self.clean_from_driver(self.__scattered_disciplines[disc])
            del self.__scattered_disciplines[disc]

    def clean_all_disciplines_from_tool(self):
        all_disc_list = self.get_all_built_disciplines()
        self.clean_from_driver(all_disc_list)

    def clean_from_driver(self, disc_list):
        """
        This method cleans the given list of children from the current discipline
        """
        self.driver.clean_children(disc_list)

    def add_scatter_discipline(self, disc, name):
        '''
        Add the discipline to the factory and to the dictionary of scattered_disciplines
        '''
        self.set_father_discipline()
        self.ee.factory.add_discipline(disc)
        if name in self.__scattered_disciplines.keys():
            self.__scattered_disciplines[name].append(disc)
        else:
            self.__scattered_disciplines.update({name: [disc]})

    def get_all_built_disciplines(self):

        return [disc for disc_list in self.__scattered_disciplines.values() for disc in disc_list]

    def get_all_built_disciplines_names(self):
        return list(self.__scattered_disciplines.keys())

    def get_all_gather_disciplines(self):
        return list(self.__gather_disciplines.values())

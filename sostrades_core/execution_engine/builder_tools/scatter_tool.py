'''
Copyright 2022 Airbus SAS
Modifications on 2023/07/17-2023/11/02 Copyright 2023 Capgemini

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
                                     'group_scenarios_under_disciplines', 'autogather']

    def __init__(self, sos_name, ee, cls_builder, map_name=None, coupling_per_scenario=True,
                 display_options=False):
        '''
        Constructor
        '''

        SosTool.__init__(self, sos_name, ee, cls_builder)

        self.map_name = map_name
        self.coupling_per_scenario = coupling_per_scenario
        self.display_options = {disp_option: False for disp_option in self.DISPLAY_OPTIONS_POSSIBILITIES}
        self.set_display_options(display_options)
        self.driver_display_value = None
        self.__scattered_disciplines = {}
        self.__gather_disciplines = {}
        self.sub_coupling_builder_dict = {}
        self.__scatter_list = None
        self.input_name = None
        self.ns_to_update = None
        self.sc_map = None

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
        MEthod that associate tool to the driver and add scatter map
        '''
        SosTool.associate_tool_to_driver(
            self, driver, cls_builder=cls_builder, associated_namespaces=associated_namespaces)

        if self.map_name is not None:
            self.sc_map = self.ee.scattermap_manager.get_build_map(self.map_name)
            self.ee.scattermap_manager.associate_disc_to_build_map(self)
            self.sc_map.configure_map(self.sub_builders)
        # get new namespace values at first config rather than at prepare_tool phase to do it only once
        self.get_values_for_namespaces_to_update()

    def prepare_tool(self):
        '''
        Prepare tool function if some data of the driver are needed to configure the tool
        '''

        super().prepare_tool()
        if self.driver.SCENARIO_DF in self.driver.get_data_in():
            instance_reference = self.driver.get_sosdisc_inputs(self.driver.INSTANCE_REFERENCE)
            scenario_df = self.driver.get_sosdisc_inputs(self.driver.SCENARIO_DF)
            # sce_df = copy.deepcopy(scenario_df)
            if instance_reference:
                scenario_df = scenario_df.append(
                    {self.driver.SELECTED_SCENARIO: True, self.driver.SCENARIO_NAME: 'ReferenceScenario'},
                    ignore_index=True)

            self.set_scatter_list(
                scenario_df[scenario_df[self.driver.SELECTED_SCENARIO] == True][
                    self.driver.SCENARIO_NAME].values.tolist())

        if self.display_options['hide_coupling_in_driver']:
            self.driver_display_value = self.driver.get_disc_display_name()

        if self.display_options['group_scenarios_under_disciplines'] or self.display_options['autogather']:
            self.flatten_subprocess = True

        if self.flatten_subprocess:
            self.coupling_per_scenario = False

    def get_values_for_namespaces_to_update(self):
        '''
        Get the values of the namespace list defined in the namespace manager
        '''
        ns_to_update_name_list = self.get_ns_to_update_name_list()

        # store ns_to_update namespace object
        self.ns_to_update = {}
        for ns_name in ns_to_update_name_list:
            if not self.flatten_subprocess:
                self.ns_to_update[ns_name] = self.ee.ns_manager.get_shared_namespace(self.driver,
                                                                                     ns_name)
            else:
                # if flatten subprocess then the father evaluator for a nested scatter is always the root coupling
                # and we should take ns_to_update of the shared_ns_dict to be consistent with father_executor name and driver_name
                self.ns_to_update[ns_name] = self.ee.ns_manager.get_ns_in_shared_ns_dict(ns_name)

    def get_dynamic_output_from_tool(self):
        '''
        Add the scatter list output name into dynamic desc_out in the behalf of the driver
        this scatter_list is depending on scenario_df configuration
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

            new_sub_names = self.clean_scattered_disciplines(
                self.__scatter_list)

            if self.display_options['autogather']:
                self.add_gather()
            # build sub_process through the factory
            for name in self.__scatter_list:
                new_name = name in new_sub_names
                ns_ids_list = self.update_namespaces(name)
                if self.coupling_per_scenario:
                    self.build_sub_coupling(
                        name, new_name, ns_ids_list)
                else:
                    self.build_child_scatter(
                        name, new_name, ns_ids_list)


    def build_sub_coupling(self, name, new_name, ns_ids_list):
        '''
        Build a coupling for each name 
        name (string) : scatter_name in the scatter_list
        new_name (bool) : True if name is a new_name in the build
        ns_ids_list (list) : The list of ns_keys that already have been updated with the scatter_name and mus tbe associated to the builder 

        1. Create the coupling with its name
        2. Add all builders to the coupling
        3. Associate new namespaces to the builder (the coupling will associate the namespace to its children)
        4. Build the coupling builder
        5 Add the builded discipline to the driver and factory
        '''
        # Call scatter map to modify the associated namespace

        if new_name:
            coupling_builder = self.driver.create_sub_builder_coupling(
                name, self.sub_builders)

            if self.display_options['hide_coupling_in_driver']:
                self.ee.ns_manager.add_display_ns_to_builder(
                    coupling_builder, self.driver_display_value)

            self.associate_namespaces_to_builder(
                coupling_builder, ns_ids_list)
            self.sub_coupling_builder_dict[name] = coupling_builder

            coupling_disc = coupling_builder.build()
            # flag the coupling so that it can be executed in parallel
            coupling_disc.is_parallel = True
            self.add_scatter_discipline(coupling_disc, name)

        else:
            coupling_disc = self.sub_coupling_builder_dict[name].build()
            # flag the coupling so that it can be executed in parallel
            coupling_disc.is_parallel = True

    def update_namespaces(self, name):
        '''
        Update all ns_to_update namespaces and the scatter namespace with the scatter_name just after the local_namespace
        Return the list of namespace keys for future builder association
        All namespaces are not added in shared_ns_dict to be transparent and only associated to the right disciplines
        '''

        ns_ids_list = []

        extra_name = f'{self.driver.sos_name}.{name}'
        after_name = self.driver.father_executor.get_disc_full_name()
        ns_list = self.ns_to_update.values()

        for ns_name, ns in self.ns_to_update.items():
            updated_value = self.ee.ns_manager.update_ns_value_with_extra_ns(
                ns.get_value(), extra_name, after_name=after_name)
            display_value = ns.get_display_value_if_exists()
            ns_id = self.ee.ns_manager.add_ns(
                ns_name, updated_value, display_value=display_value, add_in_shared_ns_dict=False, clean_existing=False)
            ns_ids_list.append(ns_id)

        # remove/clean the initial namespace values of the subprocess before they were updated
        self.ee.ns_manager.clean_all_ns_in_nslist(ns_list, clean_all_ns_with_name=False)

        return ns_ids_list

    def build_child_scatter(self, name, new_name, ns_ids_list):
        '''
        #        |_name_1
        #                |_Disc1
        #                |_Disc2
        #        |_name_2
        #                |_Disc1
        #                |_Disc2

        Build scattered disciplines directly under driver
        name (string) : scatter_name in the scatter_list
        new_name (bool) : True if name is a new_name in the build
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
            # if flatten subprocess then the discipline will be build at coupling above the driver
            # then the name of the driver must be inside the discipline name
            # else the discipline is build in the driver then no need of driver_name
            if self.flatten_subprocess:
                driver_name = self.driver.sos_name
                disc_name = f'{driver_name}.{name}.{old_builder_name}'
            else:
                disc_name = f'{name}.{old_builder_name}'

            builder.set_disc_name(disc_name)
            if new_name:
                self.associate_namespaces_to_builder(builder, ns_ids_list)
            self.set_father_discipline()
            disc = builder.build()

            self.apply_display_options(disc, name, old_builder_name)

            builder.set_disc_name(old_builder_name)
            # Add the discipline only if it is a new_name
            if new_name:
                self.add_scatter_discipline(disc, name)

    def apply_display_options(self, disc, name, old_builder_name):
        '''
        Apply the display options proposed by the driver in multiinstance mode
        1. hide_under_coupling  : Hide all the disicplines instanciated under the high level coupling
        2. group_scenarios_under_disciplines : Group All the scenario under each discipline in display treeview (the exec treeview remains the same)
        3. autogather : Add a Gather discipline which will autogather disciplines
        '''

        if self.display_options['hide_under_coupling']:
            local_ns_disc = self.ee.ns_manager.get_local_namespace(disc)
            display_value = f'{self.driver.get_disc_display_name()}.{name}'
            local_ns_disc.set_display_value(display_value)
        elif self.display_options['group_scenarios_under_disciplines']:
            local_ns_disc = self.ee.ns_manager.get_local_namespace(disc)
            display_value = f'{self.driver.get_disc_display_name()}.{old_builder_name}.{name}'
            local_ns_disc.set_display_value(display_value)
        if self.display_options['autogather']:
            if self.display_options['group_scenarios_under_disciplines']:
                gather_name = old_builder_name
            else:
                gather_name = f'{self.driver.sos_name}_gather'

            self.__gather_disciplines[gather_name].add_disc_to_config_dependency_disciplines(disc)

    def add_gather(self):
        '''
            Add gather discipline for autogather
            the gather discipline name will automatically be the name of the builder
            if display option group_scenarios_under_disciplines s activated then we want a gather per subbuilder
            if not only one gather per driver which gather automatically all outputs need flatten_subprocess option in order to gather correctly all disciplines
        '''

        if self.display_options['group_scenarios_under_disciplines']:
            for sub_builder in self.sub_builders:
                if self.flatten_subprocess:
                    gather_name = f'{self.driver.sos_name}.{sub_builder.sos_name}'
                else:
                    gather_name = sub_builder.sos_name
                if sub_builder.sos_name not in self.__gather_disciplines:
                    gather_builder = self.ee.factory.add_gather_builder(gather_name)
                    self.set_father_discipline()
                    gather_disc = gather_builder.build()
                    self.ee.factory.add_discipline(gather_disc)
                    self.__gather_disciplines[sub_builder.sos_name] = gather_disc
        else:
            gather_name = f'{self.driver.sos_name}_gather'
            if gather_name not in self.__gather_disciplines:
                gather_builder = self.ee.factory.add_gather_builder(gather_name)
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
        clean_builders_list = []
        new_sub_names = [
            name for name in sub_names if not name in self.__scattered_disciplines]
        disc_name_to_remove = [
            name for name in self.__scattered_disciplines if not name in sub_names]
        self.remove_scattered_disciplines(disc_name_to_remove)

        return new_sub_names

    def remove_scattered_disciplines(self, disc_to_remove):
        '''
        Remove a list of disciplines from the scattered_disciplines
        '''

        for disc in disc_to_remove:
            self.clean_from_driver(self.__scattered_disciplines[disc])
            if self.coupling_per_scenario:
                del self.sub_coupling_builder_dict[disc]

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
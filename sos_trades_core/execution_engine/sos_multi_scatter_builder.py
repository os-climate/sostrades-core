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
from sos_trades_core.execution_engine.sos_discipline_builder import SoSDisciplineBuilder
from sos_trades_core.execution_engine.sos_discipline_scatter import SoSDisciplineScatter
import numpy as np
from copy import deepcopy, copy
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline


class SoSMultiScatterBuilderException(Exception):
    pass


class SoSMultiScatterBuilder(SoSDisciplineBuilder):
    ''' 
    Class that build several scatter following an input_builder_dict  
    '''
    ASSOCIATED_INPUTS = 'associated_inputs'

    def __init__(self, sos_name, ee, associated_builder_list, own_map_name, child_map_name, autogather, builder_child_path=None):
        '''
        Constructor
        '''
        self.__dynamic_disciplines = {}
        SoSDisciplineBuilder.__init__(
            self, sos_name, ee)

        self.associated_builder_list = associated_builder_list
        self.activated_vb = {}
        self._builder_dict = None

        # Get the map name for the child scatter which will be builded in the
        # multiscatter builder
        self.child_map_name = child_map_name
        self.child_sc_map = ee.smaps_manager.get_build_map(
            self.child_map_name)

        # Get the map name of the multiscatter builder
        self.own_map_name = own_map_name
        self.own_sc_map = ee.smaps_manager.get_build_map(
            self.own_map_name)

        self.child_scatter_builder_list = {}
        # Build the input variable related to the mukti scatter builder
        self.build_input_builder_dict_in_inst_desc_in()
        self.autogather = autogather
        self._maturity = ''

        self.builder_child_path = builder_child_path

    def get_dynamic_disciplines(self):
        return self.__dynamic_disciplines

    def build_input_builder_dict_in_inst_desc_in(self):
        '''
        Consult the associated scatter map and adapt the inst_desc_in with the variable input_builder_dict
        '''
        input_name = self.own_sc_map.get_input_name()
        input_type = self.own_sc_map.get_input_type()
        input_ns = self.own_sc_map.get_input_ns()
        default_input = self.own_sc_map.get_default_input()
        # add ASSOCIATED_INPUTS field in map to store input whose namespace
        # need to be updated
        self.own_sc_map.map[self.ASSOCIATED_INPUTS] = []

        dict_desc_in = {input_name: {
            SoSDisciplineBuilder.TYPE: input_type,
            SoSDisciplineBuilder.VISIBILITY: SoSDisciplineBuilder.SHARED_VISIBILITY,
            SoSDisciplineBuilder.NAMESPACE: input_ns,
            SoSDisciplineBuilder.EDITABLE: True,
            SoSDisciplineBuilder.DEFAULT: default_input,
            SoSDisciplineBuilder.DATAFRAME_DESCRIPTOR: {
                'variable': ('string', None, True),
                'value': ('string', None, True)
            },
            SoSDisciplineBuilder.DATAFRAME_EDITION_LOCKED: False,
            SoSDisciplineBuilder.STRUCTURING: True
        }}
        if input_name not in self.inst_desc_in:
            self.inst_desc_in.update(dict_desc_in)

    def build(self):
        '''
        Build multiple scatters and corresponding scatter maps depending on the input builder dict
        '''

        if self._data_in is not None and self.own_sc_map.get_input_name() in self._data_in:
            input_builder_dict = self.get_sosdisc_inputs(
                self.own_sc_map.get_input_name())
            if input_builder_dict is not None:
                # Check the structure of the input builder dict
                self.check_input_builder_dict_structure(input_builder_dict)
                # Create all lists in desc_out used for child scatters
                old_activated_vb = deepcopy(self.activated_vb)
                self.create_dynamic_input_lists(input_builder_dict)
                # Clean disciplines that are not in the input builder dict
                # anymore and return new ones
                new_children = self.clean_scatter_disciplines(
                    old_activated_vb)
                # Loop on the input builder dict keys
                for name in input_builder_dict:
                    old_vb_name_list = []
                    # Set the name of all builders of each scatter with the
                    # name inside the input builder dict
                    for vb_builder in self.associated_builder_list:
                        old_vb_name = vb_builder.sos_name
                        old_vb_name_list.append(old_vb_name)
                        vb_builder.set_disc_name(
                            f'{name}{self.ee.ns_manager.NS_SEP}{old_vb_name}')
                    # If the scatter must be created :
                    if len(new_children[name]) > 0:
                        self.build_child_scatter_and_map(
                            name, new_children[name])

                        if self.builder_child_path is not None:
                            parent_builder = self.ee.factory.create_sum_builder(
                                name, self.builder_child_path['parent'])
                            self.child_scatter_builder_list[name].append(
                                parent_builder)

                    # Need to reset the builder name with the name of the child
                    # before the build of the child scatter
                    for i, vb_builder in enumerate(self.associated_builder_list):
                        vb_builder.set_disc_name(
                            f'{self.sos_name}{self.ee.ns_manager.NS_SEP}{name}{self.ee.ns_manager.NS_SEP}{old_vb_name_list[i]}')

                    # Recreate list for sub scatter id they have been modified
                    self.create_input_name_from_input_builder_dict()
                    # Build child scatter builders created before (always build
                    # all child scatter)
                    if name in self.child_scatter_builder_list:
                        for child_scatter in self.child_scatter_builder_list[name]:
                            # update ns_to_update_with_actor with actor name
                            old_ns_to_update_with_actor = {}
                            for ns_name in self.own_sc_map.get_ns_to_update_with_actor():
                                old_ns_to_update_with_actor[ns_name] = self.ee.ns_manager.get_shared_namespace(self,
                                                                                                               ns_name)
                            self.own_sc_map.update_ns(
                                old_ns_to_update_with_actor, name, self.sos_name)
                            # build child scatter
                            child_scatter_disc = child_scatter.build()
                            # Add the scatter discipline to the factory
                            self.add_dynamic_discipline(
                                child_scatter_disc)
                            # restore value of ns_to_update
                            self.ee.ns_manager.shared_ns_dict.update(
                                old_ns_to_update_with_actor)
                    # Revert the child name in the builders for the next
                    # child
                    for builder in self.associated_builder_list:
                        old_vb_name = old_vb_name_list.pop(0)
                        builder.set_disc_name(old_vb_name)

    def build_child_scatter_and_map(self, name, vb_list):
        '''
        Build a child scatter and a child map with the corresponding name
        '''
        self.ee.ns_manager.update_shared_ns_with_others_ns(self)

        for vb_builder in self.associated_builder_list:
            vb_name = vb_builder.sos_name.split('.')[-1]
            if vb_name in vb_list:
                # if actor=vb_name in vb_list=value block to build
                child_map_name = f'{self.child_sc_map.get_input_name()}_{name}_{vb_name}'
                child_map = self.child_sc_map.get_map().copy()

                if child_map_name not in self.ee.smaps_manager.build_maps_dict:
                    child_map.update({'input_name': child_map_name,
                                      'input_ns': f'ns_{name}_{vb_name}',
                                      'ns_to_update': self.own_sc_map.get_ns_to_update(),
                                      'gather_ns_out': f'ns_{name}'})
                    self.ee.smaps_manager.add_build_map(
                        child_map_name, child_map)

                if self.builder_child_path is None:
                    path_sum = None
                else:
                    path_sum = self.builder_child_path['children']

                if vb_builder.cls.__name__ == 'SoSDisciplineGather':
                    autogather = False
                else:
                    autogather = self.autogather

                child_scatter_build_list = self.ee.factory.create_multi_scatter_builder_from_list(
                    child_map_name, [vb_builder], autogather, path_sum=path_sum)

                if name in self.child_scatter_builder_list:
                    self.child_scatter_builder_list[name].extend(
                        child_scatter_build_list)
                else:
                    self.child_scatter_builder_list[name] = child_scatter_build_list

    def check_input_builder_dict_structure(self, input_builder_dict):
        '''
        Check if we have the correct structure for the value blocks dict
        '''

        for actor in input_builder_dict:
            if isinstance(input_builder_dict[actor], dict):
                for aircraft in input_builder_dict[actor]:
                    if not isinstance(input_builder_dict[actor][aircraft], list):
                        raise SoSMultiScatterBuilderException(
                            f'Values blocks for the actor {actor} and the aircraft {aircraft} must be a list of value blocks')
                    else:
                        for vb in input_builder_dict[actor][aircraft]:
                            if vb not in [builder.sos_name.split('.')[-1] for builder in self.associated_builder_list]:
                                raise SoSMultiScatterBuilderException(
                                    f'There is no value block builder called \'{vb}\' in the process')
            else:
                raise SoSMultiScatterBuilderException(
                    f'The dict of value blocks for {actor} is not correct, must be a dict')

    def create_input_name_from_input_builder_dict(self):
        ''' 
        Create the input_name from the input_builder_dict
        '''
        total_ac_list = []

        if self._data_in is not None and self.own_sc_map.get_input_name() in self._data_in:
            input_builder_dict = self.get_sosdisc_inputs(
                self.own_sc_map.get_input_name())
            if input_builder_dict is not None:

                if self.child_sc_map.get_input_name() in self._data_in:
                    for name in input_builder_dict:
                        # input_key depends on actor
                        input_key = f'{self.child_sc_map.get_input_name()}_{name}'
                        val_input_key = []
                        for key, val in input_builder_dict[name].items():
                            if len(val) > 0:
                                val_input_key.append(key)
                        if input_key in self._data_in:
                            self.ee.dm.set_data(self.get_var_full_name(
                                input_key, self._data_in), self.VALUE, val_input_key)
                        total_ac_list.extend(val_input_key)

                    total_ac_list = list(set(total_ac_list))
                    self.ee.dm.set_data(self.get_var_full_name(
                        self.child_sc_map.get_input_name(), self._data_in), self.VALUE, total_ac_list)

                # store activated vb inputs values used in maps to build vb
                # scatter
                for actor, vb_list in self.activated_vb.items():
                    for vb_name in vb_list:
                        # input_key depends on actor and value block
                        input_key = f'{self.child_sc_map.get_input_name()}_{actor}_{vb_name}'
                        if input_key in self._data_in:
                            input_value = []
                            for name, value in input_builder_dict[actor].items():
                                if vb_name in value:
                                    input_value.append(name)
                            self.ee.dm.set_data(self.get_var_full_name(
                                input_key, self._data_in), self.VALUE, input_value)

    def add_dynamic_discipline(self, disc):
        '''
        Add the discipline to the factory and to the dictionary of disciplines
        '''

        if disc.disc_id not in self.__dynamic_disciplines.keys():
            self.ee.factory.add_discipline(disc)
            self.__dynamic_disciplines.update({disc.disc_id: [disc]})

    def clean_scatter_disciplines(self, old_activated_vb):
        '''
        Clean disciplines that was scattered and are not in the scatter_list anymore
        Return the new scatter names not yet present in the list of scattered disciplines
        '''
        # compare old_activated_vb and self.activated_vb to determine newnames
        # to build
        new_names = {}
        for name, vb_list in self.activated_vb.items():
            new_names[name] = []
            if name in old_activated_vb:
                for vb in vb_list:
                    if not vb in old_activated_vb[name]:
                        new_names[name].append(vb)
            else:
                new_names[name].extend(vb_list)

        # compare old_activated_vb and self.activated_vb to determine old names
        # to remove
        names_to_remove = {}
        for name, vb_list in old_activated_vb.items():
            sub_names = []
            if name in self.activated_vb:
                for vb in vb_list:
                    if not vb in self.activated_vb[name]:
                        sub_names.append(vb)
            else:
                sub_names.extend(vb_list)
            if len(sub_names) > 0 or name not in new_names:
                names_to_remove[name] = sub_names

        disciplines_to_remove = []
        builders_to_remove = []
        remove_all_builders = False
        for name in names_to_remove:
            if set(old_activated_vb[name]) == set(names_to_remove[name]):
                # remove previous input per actor
                input_key = f'{self.child_sc_map.get_input_name()}_{name}'
                if input_key in self._data_in:
                    full_input_key = self.get_var_full_name(
                        input_key, self._data_in)
                    self.ee.dm.remove_keys(self.disc_id, [full_input_key])
                    del self._data_in[input_key]
                    del self.inst_desc_in[input_key]

                if name in self.child_scatter_builder_list:
                    builders_to_remove = copy(
                        self.child_scatter_builder_list[name])
                remove_all_builders = True

            for vb in names_to_remove[name]:
                # remove previous input per actor per value block
                input_key = f'{self.child_sc_map.get_input_name()}_{name}_{vb}'
                if input_key in self._data_in:
                    full_input_key = self.get_var_full_name(
                        input_key, self._data_in)
                    self.ee.dm.remove_keys(self.disc_id, [full_input_key])
                    del self._data_in[input_key]
                    del self.inst_desc_in[input_key]

                if not remove_all_builders:
                    builders_to_remove.extend([
                        builder for builder in self.child_scatter_builder_list[name] if builder.sos_name == f'{name}.{vb}'])

            # remove builders
            for old_builder in builders_to_remove:
                for disc in list(old_builder.discipline_dict.values()):
                    self.add_disc_to_remove_recursive(
                        disc, disciplines_to_remove)
                    del self.__dynamic_disciplines[disc.disc_id]
                self.child_scatter_builder_list[name].remove(
                    old_builder)

            if remove_all_builders and name in self.child_scatter_builder_list:
                del self.child_scatter_builder_list[name]

        # clean disciplines in current_discipline and
        # scattered_disciplines
        self.ee.factory.clean_discipline_list(disciplines_to_remove)

        return new_names

    def add_disc_to_remove_recursive(self, disc, to_remove):
        if isinstance(disc, SoSDisciplineScatter):
            if disc not in to_remove:
                to_remove.append(disc)
            scattered_discs = disc.get_scattered_disciplines()
            for disc_list in scattered_discs.values():
                for sub_disc in disc_list:
                    self.add_disc_to_remove_recursive(sub_disc, to_remove)
        else:
            if disc not in to_remove:
                to_remove.append(disc)

    def configure(self):
        '''
        Configure as a discipline and put the value of each DESC_OUT from the input builder_dict
        '''
        # first configure to add input_map in data_in
        if self._data_in == {}:
            SoSDiscipline.configure(self)
        elif not all([input in self._data_in.keys() for input in self.inst_desc_in.keys()]):
            # add sub level input map in data_in, created during build step
            self.reload_io()

        input_value = self.get_sosdisc_inputs(self.own_sc_map.get_input_name())
        if input_value is not None and self.child_scatter_builder_list.keys() == input_value.keys():
            if self.check_structuring_variables_changes():
                # store new values of structuring variables
                self.set_structuring_variables_values()
            self.create_input_name_from_input_builder_dict()
            self._builder_dict = deepcopy(input_value)

    def create_dynamic_input_lists(self, input_builder_dict):
        '''
        Create the dynamic DESC_OUT for input names of the own scatter map AND for all child scatter maps created in the build
        '''
        input_type = self.child_sc_map.get_input_type()
        input_ns = self.child_sc_map.get_input_ns()
        local_ns = self.ee.ns_manager.get_local_namespace(self).get_value()

        if not self.child_sc_map.get_input_name() in self._data_in:
            total_input = {self.child_sc_map.get_input_name(): {
                SoSDisciplineBuilder.TYPE: input_type, SoSDisciplineBuilder.VISIBILITY: SoSDisciplineBuilder.SHARED_VISIBILITY, SoSDisciplineBuilder.NAMESPACE: input_ns, SoSDisciplineBuilder.EDITABLE: False, SoSDisciplineBuilder.USER_LEVEL: 3}}
            self.inst_desc_in.update(total_input)
            self.own_sc_map.map[self.ASSOCIATED_INPUTS].append(
                self.child_sc_map.get_input_name())
        else:
            self.ee.dm.set_data(self.get_var_full_name(
                self.child_sc_map.get_input_name(), self._data_in), self.EDITABLE, False)

        # add input per actor in inst_desc_in
        for key in input_builder_dict:
            # Comment in the case of simple multiscenario need to update the ns_actor for the new scenario
            # if f'ns_{key}' not in
            # self.ee.ns_manager.get_disc_others_ns(self):
            self.ee.ns_manager.add_ns(
                f'ns_{key}', f'{local_ns}.{key}')
            self.ee.ns_manager.update_others_ns_with_shared_ns(
                self, f'ns_{key}')
            input_key = f'{self.child_sc_map.get_input_name()}_{key}'
            if not input_key in self._data_in:
                desc_in_key = {input_key: {
                    SoSDisciplineBuilder.TYPE: input_type, SoSDisciplineBuilder.VISIBILITY: SoSDisciplineBuilder.SHARED_VISIBILITY, SoSDisciplineBuilder.NAMESPACE: f'ns_{key}', SoSDisciplineBuilder.EDITABLE: False, SoSDisciplineBuilder.USER_LEVEL: 3}}
                self.inst_desc_in.update(desc_in_key)
            else:
                self.ee.dm.set_data(self.get_var_full_name(
                    input_key, self._data_in), self.EDITABLE, False)

        # determine activated value blocks per actor
        activated_vb = {}
        for actor, vb_actor in input_builder_dict.items():
            activated_vb[actor] = list(
                set(np.concatenate(list(vb_actor.values()))))
        self.activated_vb = activated_vb

        # add dynamic input per actor per value block for activated value
        # blocks
        for actor, act_vb in activated_vb.items():
            for vb in act_vb:
                if f'ns_{actor}_{vb}' not in self.ee.ns_manager.get_disc_others_ns(self):
                    builder_name = [
                        builder.sos_name for builder in self.associated_builder_list if vb in builder.sos_name][0]
                    self.ee.ns_manager.add_ns(
                        f'ns_{actor}_{vb}', f'{local_ns}.{actor}.{builder_name}')
                    self.ee.ns_manager.update_others_ns_with_shared_ns(
                        self, f'ns_{actor}_{vb}')
                input_key = f'{self.child_sc_map.get_input_name()}_{actor}_{vb}'
                if not input_key in self._data_in:
                    desc_in_key = {input_key: {
                        SoSDisciplineBuilder.TYPE: input_type, SoSDisciplineBuilder.VISIBILITY: SoSDisciplineBuilder.SHARED_VISIBILITY, SoSDisciplineBuilder.NAMESPACE: f'ns_{actor}_{vb}', SoSDisciplineBuilder.EDITABLE: False, SoSDisciplineBuilder.USER_LEVEL: 3}}
                    self.inst_desc_in.update(desc_in_key)
                else:
                    self.ee.dm.set_data(self.get_var_full_name(
                        input_key, self._data_in), self.EDITABLE, False)

    def run(self):
        '''
        This discipline is only a builder : no run needed 
        It must be a discipline to host the input_builder_dict
        '''
        pass

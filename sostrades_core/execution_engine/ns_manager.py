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
from copy import copy, deepcopy

from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.execution_engine.namespace import Namespace

IO_TYPE_IN = ProxyDiscipline.IO_TYPE_IN
IO_TYPE_OUT = ProxyDiscipline.IO_TYPE_OUT
SHARED_VISIBILITY = ProxyDiscipline.SHARED_VISIBILITY
LOCAL_VISIBILITY = ProxyDiscipline.LOCAL_VISIBILITY
INTERNAL_VISIBILITY = ProxyDiscipline.INTERNAL_VISIBILITY
NS_SEP = '.'


class NamespaceManager:
    '''
    Specification: NamespaceManager allows to manage namespaces for disciplines data
    '''
    NS_SEP = '.'
    NS_NAME_SEPARATOR = Namespace.NS_NAME_SEPARATOR

    def __init__(self, name, ee):
        '''
            Constructor
        '''
        self.name = name  # old habit

        self.ee = ee
        #-- List where all namespaces are gathered (of all disciplines)
        self.ns_list = []
        # Dict with key = ns name and value = ns value just for performances
        self.all_ns_dict = {}
        # Dict of shared namespaces which fills the others_ns key of the
        # disc_ns_dict
        self.__shared_ns_dict = {}

        # Dict of local namespaces which fills the local_ns key of the
        # disc_ns_dict
        self.__local_ns_dict = {}
        #-- current disciplinary name space
        self.current_disc_ns = None

        # disc_ns_dict hosts local_ns and other_ns for each discipline in the
        # exec engine (key is the instanciated discipline)
        self.__disc_ns_dict = {}

        self.display_ns_dict = {}
        self.logger = ee.logger.getChild("NamespaceManager")

        # List of dict with extra_ns and ater_name infos for local namespace
        # update
        self.extra_ns_local = []
        self.ns_object_map = {}

        self.database_activated = False 

    @staticmethod
    def compose_ns(args):
        ''' concatenate list of string items as namespace-like '''
        if not isinstance(args, list) and not isinstance(args, tuple):
            args = [args]
        if None in args:
            args.remove(None)
        if len(args) == 0:
            raise NamespaceManagerException(
                f'no argument given to NamespaceManager.compose_ns')
        elif len(args) == 1:
            return args[0]
        else:
            return NS_SEP.join(args)

    @property
    def shared_ns_dict(self):
        '''
        Dict of shared namespaces which fills the others_ns key of the disc_ns_dict
        '''
        return self.__shared_ns_dict

    @property
    def local_ns_dict(self):
        '''
        Dict of local namespaces which fills the local_ns key of the disc_ns_dict
        '''
        return self.__local_ns_dict

    @property
    def disc_ns_dict(self):
        '''
        disc_ns_dict hosts local_ns and other_ns for each discipline in the
        exec engine (key is the instanciated discipline)
        '''
        return self.__disc_ns_dict

    #-- Data name space methods
    def add_ns_def(self, ns_info, get_from_database = False):
        ''' 
        add multiple namespaces to the namespace_manager 
        ns_info is a dict with the key equals to the name and the value is a namespace to add
        '''
        ns_ids = []
        for key, value in ns_info.items():
            ns_id = self.add_ns(key, value, get_from_database=get_from_database)
            ns_ids.append(ns_id)
        if get_from_database: 
            self.database_activated = True

        return ns_ids

    def add_ns(self, name, ns_value, display_value=None, add_in_shared_ns_dict=True, get_from_database = False):
        '''
        add namespace to namespace manager
        WARNING: Do not use to update namespace values
        '''

        # if the couple (name,value) already exists do not create another
        # object take the one that exists
        ns_id = f'{name}{self.NS_NAME_SEPARATOR}{ns_value}'

        if ns_id in self.all_ns_dict:
            ns = self.all_ns_dict[ns_id]
            ns.get_from_database = get_from_database

        # else we create a new object and store it in all_ns_dict
        else:
            ns = Namespace(name, ns_value, display_value, get_from_database)
            #-- add in the list if created
            self.ns_list.append(ns)
            self.all_ns_dict[ns.get_ns_id()] = ns
        # This shared_ns_dict delete the namespace if already exist: new one
        # has priority
        if add_in_shared_ns_dict:
            self.shared_ns_dict[name] = ns
        self.ns_object_map[id(ns)] = ns

        return ns.get_ns_id()

    def add_display_ns_to_builder(self, disc_builder, display_value):
        '''
        Associate a display value to a builder
        when the builder will build the disciplines it will automatically add the display value to the local namespace
        '''
        self.display_ns_dict[disc_builder] = display_value
        self.associate_display_values_to_new_local_namespaces(disc_builder)

    def delete_display_ns_in_builder(self, disc_builder):
        '''
        Delete a builder in the display_ns_dict
        '''
        del self.display_ns_dict[disc_builder]

    def add_display_ns_to_builder_list(self, builder_list, display_value):
        '''
        Associate a display value to a builder_list
        when the builder will build the disciplines it will automatically add the display value to the local namespace
        '''
        for builder in builder_list:
            self.add_display_ns_to_builder(builder, display_value)

    def associate_display_values_to_new_local_namespaces(self, disc_builder):
        if disc_builder in self.display_ns_dict:
            display_value = self.display_ns_dict[disc_builder]
            for disc in disc_builder.discipline_dict.values():
                if disc in self.local_ns_dict:
                    self.local_ns_dict[disc].set_display_value(
                        display_value)

    def get_all_namespace_with_name(self, name):
        '''
        Get all namespaces with same name
        '''
        ns_list = []
        for namespace in self.ns_list:
            if namespace.name == name:
                ns_list.append(namespace)

        return ns_list

    def get_shared_ns_dict(self):
        '''
        Get a deepcopy of the shared_ns_dict
        '''
        return copy(self.shared_ns_dict)

    def get_ns_in_shared_ns_dict(self, ns_name):
        '''
        Get a deepcopy of the shared_ns_dict
        '''
        if ns_name not in self.shared_ns_dict:
            raise Exception(
                f'The namespace {ns_name} is not defined in the namespace manager')
        else:
            return self.shared_ns_dict[ns_name]
    #-- Disciplinary name space management

    def reset_current_disc_ns(self):
        '''
        Reset the current_disc_ns to None
        '''
        self.current_disc_ns = None

    def set_current_disc_ns(self, disc_ns):
        '''
        Set directly the current_disc_ns with the argument disc_ns
        '''
        self.current_disc_ns = disc_ns

        # add extra_ns to current_disc_ns
        for extra_local in self.extra_ns_local:
            extra_ns = extra_local['extra_ns']
            after_name = extra_local['after_name']
            if after_name is None:
                if self.current_disc_ns is None:
                    self.current_disc_ns = extra_ns
                elif extra_ns not in self.current_disc_ns:

                    self.current_disc_ns = self.compose_ns([extra_ns,
                                                            self.current_disc_ns])
            else:
                if self.current_disc_ns is not None:
                    if extra_ns not in self.current_disc_ns and f'{after_name}' in self.current_disc_ns:
                        current_disc_ns_split = self.current_disc_ns.split(
                            self.NS_SEP)
                        new_ns_value_split = []
                        for item in current_disc_ns_split:
                            new_ns_value_split.append(item)
                            if item == after_name:
                                new_ns_value_split.append(extra_ns)
                        self.current_disc_ns = self.compose_ns(
                            new_ns_value_split)

    def update_shared_ns_with_others_ns(self, disc):
        '''
        Update shared_ns_dict with others_ns in disc_nd_cit[disc]
        '''
        self.shared_ns_dict.update(
            self.get_disc_others_ns(disc))

    def update_others_ns_with_shared_ns(self, disc, ns_name):
        '''
        Update ns in others_ns with ns in shared_ns_dict 
        '''
        self.get_disc_others_ns(disc).update(
            {ns_name: self.shared_ns_dict[ns_name]})

    def change_disc_ns(self, disc_name):
        '''
        Modify the current_disc namespace by adding disc_name at the end of the old one
        if disc_name is .. then suppress the last part of the namespace splitted by dots
        '''
        if disc_name == '..':
            if self.current_disc_ns is not None:
                splitted_disc_ns = self.current_disc_ns.split(self.NS_SEP)
                if len(splitted_disc_ns) > 1:
                    self.current_disc_ns = self.compose_ns(
                        splitted_disc_ns[:-1])
                else:
                    self.current_disc_ns = None
        else:
            if self.current_disc_ns is None:
                self.current_disc_ns = disc_name
            else:
                self.current_disc_ns = self.compose_ns(
                    [self.current_disc_ns, disc_name])
        return self.current_disc_ns

    def add_dependencies_to_shared_namespace(self, disc, shared_namespace_list):
        '''
        Add the discipline disc to the dependency list of each shared
        namespace in shared_namespace_list
        '''

        for namespace in shared_namespace_list:
            if namespace in self.shared_ns_dict:
                self.shared_ns_dict[namespace].add_dependency(disc.disc_id)

    def create_disc_ns_info(self, disc):
        '''
        -Create the namespace info dict disc_ns_info for
        the current discipline disc directly in the namespace_manager
        - Add this disc_ns_info to the namespace manager info dict
        - Collect all shared namepace used in this discipline
        - Add this discipline to the dependency disc lit of each found namespace
        '''

        local_ns = self.create_local_namespace(disc)

        others_ns = self.get_associated_ns(disc)

        disc_ns_info = {'local_ns': local_ns,
                        'others_ns': others_ns}
        self.add_disc_ns_info(disc, disc_ns_info)

    def get_associated_ns(self, disc):
        '''
        Get the others_ns by default from shared_ns_dict
        IF the discipline has some associated namespaces then the others_ns dict is build in priority with these namespaces
        for other namespaces not "associated" we pick namespaces from shared_ns_dict
        '''
        shared_ns_dict = self.get_shared_ns_dict()
        if len(disc.associated_namespaces) == 0:
            others_ns = shared_ns_dict
        else:
            get_ns_names = [
                self.all_ns_dict[ns].name for ns in disc.associated_namespaces]
            if len(get_ns_names) != len(set(get_ns_names)):
                raise Exception(
                    f'There is two namespaces with the same name in the associated namespace list of {disc.sos_name}')
            others_ns = {
                self.all_ns_dict[ns].name: self.all_ns_dict[ns] for ns in disc.associated_namespaces}
            # FIX to wait all process modifs
            # add namespaces present in shared_ns_dict and not in associated_ns
            for shared_ns_name, shared_ns in shared_ns_dict.items():
                if shared_ns_name not in others_ns:
                    others_ns[shared_ns_name] = shared_ns
        return others_ns

    def compose_local_namespace_value(self, disc):
        return self.compose_ns([self.current_disc_ns, disc.sos_name])

    def create_local_namespace(self, disc):
        '''
         Create a namespace object for the local namespace
        '''
        local_ns_value = self.compose_local_namespace_value(disc)

        local_ns = Namespace(disc.sos_name, local_ns_value)

        self.local_ns_dict[disc] = local_ns
        self.ns_object_map[id(local_ns)] = local_ns

        return local_ns

    def remove_dependencies_after_disc_deletion(self, disc, disc_id=None):
        '''
        Remove dependencies of deleted disc for all namespaces
        '''
        others_ns = copy(self.get_disc_others_ns(disc))
        if disc_id is None:
            disc_id = disc.get_disc_id_from_namespace()
        for ns in others_ns.values():
            ns.remove_dependency(disc_id)
        del self.disc_ns_dict[disc]

    def clean_ns_without_dependencies(self):
        '''
        Delete namespaces without dependency in ns_list
        '''
        for ns in self.ns_list:
            dependendy_disc_id_list = ns.get_dependency_disc_list()
            dependency_disc_list = [self.ee.dm.get_discipline(
                disc_id) for disc_id in dependendy_disc_id_list]
            if len(list(filter(None, dependency_disc_list))) == 0:
                self.ns_list.remove(ns)
                del self.all_ns_dict[f'{ns.name}{self.NS_NAME_SEPARATOR}{ns.value}']
                del self.shared_ns_dict[ns.name]

    def add_disc_ns_info(self, pt, disc_ns_info):
        '''
        Add disc namespace informations to the full namespace dict 
        pt : disc which needs the ns_dict
        '''
        # If the ns_dict already exists we must update the ohters_ns with all namespace rules
        # in order to not destroy old rules
        if pt in self.disc_ns_dict:
            others_ns = self.get_disc_others_ns(pt)
            if 'others_ns' in disc_ns_info:
                others_ns.update(disc_ns_info['others_ns'])
            self.disc_ns_dict[pt] = {'local_ns': self.get_local_namespace(pt),
                                     'others_ns': others_ns}
        else:
            self.disc_ns_dict[pt] = disc_ns_info

    def add_new_shared_ns_for_disc(self, disc, shared_ns):

        if disc not in self.disc_ns_dict:
            raise Exception(f'The discipline {disc} has not been created')
        else:
            self.ns_object_map[id(shared_ns)] = shared_ns
            self.disc_ns_dict[disc]['others_ns'].update({shared_ns.name: shared_ns})

    def get_disc_ns_info(self, disc):
        '''
        get the disc_ns_info of a specified discipline disc 
        The key is the signature of the instance
        '''
        return self.disc_ns_dict[disc]

    def get_disc_others_ns(self, disc):
        '''
        Get the others namespace dict of the specified discipline disc
        '''
        return self.get_disc_ns_info(disc)['others_ns']

    def check_namespace_name_in_ns_manager(self, disc, var_ns):

        return var_ns in self.get_disc_others_ns(disc)

    def get_shared_namespace_value(self, disc, var_ns):
        '''
        Return the value of the shared_namespace linked to var_ns for the discipline disc
        '''
        if not self.check_namespace_name_in_ns_manager(disc, var_ns):
            raise Exception(
                f'The namespace {var_ns} is missing for the discipline {disc.sos_name}')
        return self.get_disc_others_ns(disc)[var_ns].get_value()

    def get_shared_namespace(self, disc, var_ns):
        '''
        Return the shared_namespace linked to var_ns for the discipline disc
        '''
        if not self.check_namespace_name_in_ns_manager(disc, var_ns):
            raise Exception(
                f'The namespace {var_ns} is missing for the discipline {disc.sos_name}')
        return self.get_disc_others_ns(disc)[var_ns]

    def get_local_namespace_value(self, disc):
        '''
        Return the local_namespace linked to the discipline disc
        '''
        return self.disc_ns_dict[disc]['local_ns'].get_value()

#     def get_display_namespace_value(self, disc):
#         '''
#         Return the display_namespace linked to the discipline disc
#         '''
#         if disc.father_builder in self.display_ns_dict:
#             return self.display_ns_dict[disc.father_builder]
#
#         elif disc.father_builder not in self.display_ns_dict and disc in self.disc_ns_dict:
#             return self.get_local_namespace_value(disc)
#         else:
#             return None

    def get_display_namespace_value(self, disc):
        return self.disc_ns_dict[disc]['local_ns'].get_display_value()

    def get_local_namespace(self, disc):
        '''
        Return the local_namespace linked to the discipline disc
        '''
        return self.disc_ns_dict[disc]['local_ns']

    def get_namespaced_variable(self, disc, var_name, io_type):
        '''
        Get the complete namespace of a variable using NS_REFERENCE and VAR_NAME
        '''
        data_io_var = disc.get_data_io_from_key(
            io_type, var_name)
        complete_var_name = data_io_var[ProxyDiscipline.VAR_NAME]

        ns_value = data_io_var[ProxyDiscipline.NS_REFERENCE].get_value()
        result = self.compose_ns([ns_value, complete_var_name])

        return result

    def get_display_variable(self, disc, var_name, io_type, exec_display=False):
        '''
        Get the complete namespace of a variable using NS_REFERENCE and VAR_NAME
        '''
        data_io_var = disc.get_data_io_from_key(
            io_type, var_name)
        complete_var_name = data_io_var[ProxyDiscipline.VAR_NAME]

        ns_value = data_io_var[ProxyDiscipline.NS_REFERENCE].get_display_value(
        )
        result = self.compose_ns([ns_value, complete_var_name])

        return result

    def ns_tuple_to_full_name(self, ns_tuple):
        """
        get variable full name from a tuple('var_name', id(ns_ref))
        """
        if isinstance(ns_tuple[0], tuple):
            var_name = ns_tuple[0][0]
        else:
            var_name = ns_tuple[0]
        ns_reference = self.ns_object_map[ns_tuple[1]]
        return self.compose_ns([ns_reference.value, var_name])

    def update_namespace_list_with_extra_ns(self, extra_ns, after_name=None, namespace_list=None):
        '''
        Update the value of a list of namespaces with an extra namespace placed behind after_name
        '''
        ns_ids = []
        if namespace_list is None:
            namespace_list = list(self.shared_ns_dict.values())
        for ns in deepcopy(namespace_list):
            ns_id = self.__update_namespace_with_extra_ns(
                ns, extra_ns, after_name)
            ns_ids.append(ns_id)

        return ns_ids

    def update_all_shared_namespaces_by_name(self, extra_ns, shared_ns_name, after_name=None):
        '''
        Update all shared namespaces named shared_ns_name with extra_namespace
        '''
        for namespace in deepcopy(self.ns_list):
            if namespace.name == shared_ns_name:
                self.__update_namespace_with_extra_ns(
                    namespace, extra_ns, after_name)

    def __update_namespace_with_extra_ns(self, old_ns_object, extra_ns, after_name=None):
        '''
        Update the value of old_ns_object with an extra namespace which will be placed just after the variable after_name
        if after is the name of the discipline then we do not add the extra namespace
        '''

        old_ns_value = old_ns_object.get_value()

        new_ns_value = self.update_ns_value_with_extra_ns(
            old_ns_value, extra_ns, after_name)

        # Add a new namespace (o or not if it exists already) but NEVER update
        # the value of a namespace without modifying the ordering of the
        # ns_manager
        ns_id = self.add_ns(old_ns_object.name, new_ns_value)
        # old_ns_object.update_value(new_ns_value)
        return ns_id

    def update_ns_value_with_extra_ns(self, ns_value, extra_ns, after_name=None):
        '''
        Add extra_ns in ns_value based on location in current_disc_ns
        if after_name is not given try to find it with the current_disc_ns
        '''
        if after_name is None:
            new_ns_value = self.compose_ns([extra_ns,
                                            ns_value])
        elif f'{after_name}' in ns_value:
            old_ns_value_split = ns_value.split(self.NS_SEP)
            new_ns_value_split = []
            for item in old_ns_value_split:
                new_ns_value_split.append(item)
                if item == after_name.split('.')[-1]:
                    new_ns_value_split.append(extra_ns)
            new_ns_value = self.compose_ns(
                new_ns_value_split)
        else:
            new_ns_value = self.compose_ns([ns_value, extra_ns])

        return new_ns_value

    def modify_all_local_namespaces_with_study_name(self, study_name):

        for ns_dict in self.disc_ns_dict.values():
            old_local_ns_value = ns_dict['local_ns'].get_value()

            ns_dict['local_ns'].update_value(
                old_local_ns_value.replace(self.ee.study_name, study_name))

    def set_ns_database_location(self, database_location):
        self.database_location = database_location

class NamespaceManagerException(Exception):
    pass

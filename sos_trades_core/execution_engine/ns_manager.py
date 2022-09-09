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
from copy import copy

from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from sos_trades_core.execution_engine.namespace import Namespace
from sos_trades_core.api import get_sos_logger

IO_TYPE_IN = SoSDiscipline.IO_TYPE_IN
IO_TYPE_OUT = SoSDiscipline.IO_TYPE_OUT
SHARED_VISIBILITY = SoSDiscipline.SHARED_VISIBILITY
LOCAL_VISIBILITY = SoSDiscipline.LOCAL_VISIBILITY
INTERNAL_VISIBILITY = SoSDiscipline.INTERNAL_VISIBILITY
NS_SEP = '.'


class NamespaceManager:
    '''
    Specification: NamespaceManager allows to manage namespaces for disciplines data
    '''
    NS_SEP = '.'

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
        self.logger = get_sos_logger(f'{self.ee.logger.name}.NamespaceManager')

        # List of dict with extra_ns and ater_name infos for local namespace
        # update
        self.extra_ns_local = []

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
    def add_ns_def(self, ns_info, overwrite_value=False):
        ''' 
        add multiple namespaces to the namespace_manager 
        ns_info is a dict with the key equals to the name and the value is a namespace to add
        '''
        for key, value in ns_info.items():
            self.add_ns(key, value)

    def add_ns(self, name, ns_value, overwrite_value=False):
        '''
        add namespace to namespace manager
        WARNING: Do not use to update namespace values
        '''
        ns = None
        if f'{name}__{ns_value}' in self.all_ns_dict:
            ns = self.all_ns_dict[f'{name}__{ns_value}']
            if overwrite_value:
                ns.value = ns_value

        # -- check if name and value
#         found = False
#         for ns_obj in self.ns_list:
#             if ns_obj.name == name and ns_obj.value == ns_value:
#                 # -- found an already existing namespace
#                 ns = ns_obj
#                 found = True
#                 break
            # else a ns already exists but with different value, continue

        # -- else generate
        if ns is None:
            ns = Namespace(name, ns_value)
            #-- add in the list if created
            self.ns_list.append(ns)
            self.all_ns_dict[f'{name}__{ns_value}'] = ns

            ns.value = ns_value
        self.shared_ns_dict[name] = ns

        return ns

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
        disc_ns_info = {'local_ns': local_ns,
                        'others_ns': self.get_shared_ns_dict()}
        self.add_disc_ns_info(disc, disc_ns_info)

    def create_local_namespace(self, disc):
        '''
         Create a namespace object for the local namespace
        '''
        local_ns_value = self.compose_ns([self.current_disc_ns, disc.sos_name])
        local_ns = Namespace(disc.sos_name, local_ns_value)

        self.local_ns_dict[disc] = local_ns

        return local_ns

    def remove_dependencies_after_disc_deletion(self, disc, disc_id=None):
        '''
        Remove dependencies of deleted disc for all namespaces
        '''
        others_ns = copy(self.get_disc_others_ns(disc))
        if disc_id is None:
            disc_id = disc.get_disc_id_from_namespace()
        for ns_name, ns in others_ns.items():
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
                del self.all_ns_dict[f'{ns.name}__{ns.value}']
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
        complete_var_name = data_io_var[SoSDiscipline.VAR_NAME]

        ns_value = data_io_var[SoSDiscipline.NS_REFERENCE].get_value()
        result = self.compose_ns([ns_value, complete_var_name])

        return result

    def update_namespace_with_extra_ns(self, old_ns_object, extra_ns, after_name=None):
        '''
        Update the value of old_ns_object with an extra namespace which will be placed just after the variable after_name
        if after is the name of the discipline then we do not add the extra namespace
        '''

        old_ns_value = old_ns_object.get_value()

        if after_name is None:
            new_ns_value = self.compose_ns([extra_ns,
                                            old_ns_value])
            old_ns_object.update_value(new_ns_value)
        else:
            if f'{after_name}' in old_ns_value:
                old_ns_value_split = old_ns_value.split(self.NS_SEP)
                new_ns_value_split = []
                for item in old_ns_value_split:
                    new_ns_value_split.append(item)
                    if item == after_name.replace(self.NS_SEP, ''):
                        new_ns_value_split.append(extra_ns)
                new_ns_value = self.compose_ns(
                    new_ns_value_split)
                old_ns_object.update_value(new_ns_value)

        return old_ns_object

    def update_namespace_list_with_extra_ns(self, extra_ns, after_name=None, namespace_list=None):
        '''
        Update the value of a list of namespaces with an extra namespace placed behind after_name
        '''
        if namespace_list is None:
            namespace_list = self.ns_list
        for ns in namespace_list:
            self.update_namespace_with_extra_ns(ns, extra_ns, after_name)

    def update_all_shared_namespaces_by_name(self, extra_ns, shared_ns_name, after_name=None):
        '''
        Update all shared namespaces named shared_ns_name with extra_namespace
        '''
        for namespace in self.ns_list:
            if namespace.name == shared_ns_name:
                self.update_namespace_with_extra_ns(
                    namespace, extra_ns, after_name)

                self.ee.dm.generate_data_id_map()

    def update_ns_value_with_extra_ns(self, ns_value, extra_ns, after_name=None):
        '''
        Add extra_ns in ns_value based on location in current_disc_ns
        if after_name is not given try to find it with the current_disc_ns
        '''
        if after_name is None:
            return self.compose_ns([extra_ns,
                                    ns_value])
        else:
            if f'{after_name}' in ns_value:
                old_ns_value_split = ns_value.split(self.NS_SEP)
                new_ns_value_split = []
                for item in old_ns_value_split:
                    new_ns_value_split.append(item)
                    if item == after_name.split('.')[-1]:
                        new_ns_value_split.append(extra_ns)
                return self.compose_ns(
                    new_ns_value_split)
            else:
                return self.compose_ns([ns_value, extra_ns])

    def modify_all_local_namespaces_with_study_name(self, study_name):

        for ns_dict in self.disc_ns_dict.values():
            old_local_ns_value = ns_dict['local_ns'].get_value()

            ns_dict['local_ns'].update_value(
                old_local_ns_value.replace(self.ee.study_name, study_name))


class NamespaceManagerException(Exception):
    pass

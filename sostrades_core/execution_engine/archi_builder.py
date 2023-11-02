'''
Copyright 2022 Airbus SAS
Modifications on 2023/05/12-2023/11/02 Copyright 2023 Capgemini

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
from copy import deepcopy, copy
from importlib import import_module

import numpy as np
import pandas as pd

from sostrades_core.execution_engine.proxy_discipline_builder import ProxyDisciplineBuilder
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline


class ArchiBuilderException(Exception):
    pass


class ArchiBuilder(ProxyDisciplineBuilder):
    """
    Class that build several disciplines following a specific architecture
    """

    # ontology information
    _ontology_data = {
        'label': 'Core Architecture Builder Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-money-bill-alt fa-fw',
        'version': '',
    }
    PARENT = 'Parent'
    CURRENT = 'Current'
    BUILDER_TYPE = 'Type'
    ACTION = 'Action'
    ACTIVATION = 'Activation'
    ARCHI_COLUMNS = [PARENT, CURRENT, BUILDER_TYPE, ACTION, ACTIVATION]
    ROOT_NODE = '@root_node@'
    POSSIBLE_ACTIONS = {
        'standard': 'standard',
        'scatter': ('scatter', 'var_name', 'builder_class', 'builder_first_scatter'),
        'architecture': ('architecture', 'architecture_df'),
        'scatter_architecture': (
            'scatter_architecture',
            'var_name',
            'builder_class',
            'architecture_df',
        ),
    }
    SELECTED_SCENARIO = 'selected_scenario'
    SCENARIO_NAME = 'scenario_name'
    ACTIVATION_DF = 'activation_df'

    DEFAULT_VB_FOLDER_LIST = ['sostrades_core.sos_wrapping']

    def __init__(self, sos_name, ee, architecture_df, cls_builder=None, associated_namespaces=None,
                 custom_vb_folder_list=None):
        """
        Constructor
        """
        super().__init__(sos_name, ee, associated_namespaces=associated_namespaces)

        self.children_dict = {}
        self.archi_disciplines = {}
        self.activated_builders = {}

        self.activation_dict = {}
        self.default_activation_df = None
        self.default_df_descriptor = None
        self.activation_columns = []
        self.activation_file_name_dict = {}
        self.vb_folder_list = []
        self.special_builders = {}
        self.driver_input_to_fill = {}

        # set the value block folder list as the custom one given as parameter
        full_vb_folder_list = []
        if custom_vb_folder_list is not None:
            full_vb_folder_list = custom_vb_folder_list

        # add the default folder for value block after the custom one if it
        # exists
        full_vb_folder_list.extend(self.DEFAULT_VB_FOLDER_LIST)

        for vb_folder in full_vb_folder_list:
            try:
                import_module(vb_folder)
                self.vb_folder_list.append(vb_folder)
            except:
                pass

        self.architecture_df = self.build_architecture_df(
            deepcopy(architecture_df))

        archi_name = self.sos_name.split('.')[-1]
        self.builder_dict, self.activation_dict = self.builder_dict_from_architecture(
            self.architecture_df, archi_name
        )

        self.get_children_list_by_vb(self.builder_dict)

    def prepare_execution(self):
        pass

    @property
    def status(self):  # type: (...) -> str
        """
        The status of the discipline, to be retrieved from the GEMSEO object after configuration.
        """

        return self.father_executor.status

    @status.setter
    def status(self, status):
        """
        setter of status
        """
        self._update_status_dm(status)

    def build_architecture_df(self, arch_df):
        """
        This method aims at building the architecture dataframe used by the archibuilder
        If there is no need to build a root node, it simply returns the architecture_df given as parameter
        If architecture_df  contains the root_node, it will be modified to
        add the root node among current nodes and set it as the parent of current nodes
        having none as parent
        """

        current_nodes = list(arch_df[self.CURRENT])

        if self.ROOT_NODE in current_nodes:
            for index, row in arch_df.iterrows():
                if (row[self.PARENT] is None) and (row[self.CURRENT] != self.ROOT_NODE):
                    row[self.PARENT] = self.sos_name
                    arch_df.loc[index, self.PARENT] = self.sos_name

                if row[self.CURRENT] == self.ROOT_NODE:
                    row[self.CURRENT] = self.sos_name
                    arch_df.loc[index, self.CURRENT] = self.sos_name

        return arch_df

    def builder_dict_from_architecture(self, archi_df, archi_parent):
        """
        Build initial builder_dict and activation_dict by reading architecture_df input
        """
        self.check_architecture(archi_df)
        activation_dict = self.create_activation_df(archi_df)
        builder_dict, activation_dict = self.create_vb_disc_namespaces_and_builders(
            archi_df, activation_dict, archi_parent
        )

        return builder_dict, activation_dict

    def check_architecture(self, archi_df):
        """
        Check the architecture dataframe to see if it is possible to build it
        """
        if archi_df.columns.tolist() != self.ARCHI_COLUMNS:
            raise ArchiBuilderException(
                f'The architecture dataframe must have 5 columns named : {self.ARCHI_COLUMNS}'
            )

        if not archi_df[self.ACTIVATION].dtype == 'bool':
            raise ArchiBuilderException(
                f'The architecture dataframe must contains bouleans in Activation column'
            )

        for action in archi_df[self.ACTION].values:
            if isinstance(action, str):
                if action != 'standard':
                    raise ArchiBuilderException(
                        f'Invalid Action in architecture dataframe. Action must be among: {list(self.POSSIBLE_ACTIONS.values())}'
                    )
            elif isinstance(action, tuple):
                if action[0] in self.POSSIBLE_ACTIONS:
                    if action[0] == 'scatter':
                        if len(action) not in [3, 4]:
                            possible_actions_basic_tuple = copy(
                                list(self.POSSIBLE_ACTIONS[action[0]])
                            )
                            possible_actions_basic_tuple.pop(-1)
                            raise ArchiBuilderException(
                                f'Invalid Action: {action}. The action \'{action[0]}\' must be defined by: {self.POSSIBLE_ACTIONS[action[0]]} or {tuple(possible_actions_basic_tuple)}'
                            )
                    elif len(action) != len(self.POSSIBLE_ACTIONS[action[0]]):
                        raise ArchiBuilderException(
                            f'Invalid Action: {action}. The action \'{action[0]}\' must be defined by: {self.POSSIBLE_ACTIONS[action[0]]}'
                        )
                else:
                    raise ArchiBuilderException(
                        f'Invalid Action in architecture dataframe. Action must be among: {list(self.POSSIBLE_ACTIONS.values())}'
                    )

    def check_activation_df(self):
        """
        Check the activation dataframe to see if possible values and types are respected
        """
        if self.ACTIVATION_DF in self.get_data_in():
            activation_df = self.get_sosdisc_inputs(self.ACTIVATION_DF)
            # chekc if sub architectures are built and activation_df has been
            # modified
            if (
                activation_df is not None
                and not activation_df.equals(
                    self._structuring_variables[self.ACTIVATION_DF]
                )
                and not activation_df.equals(self.default_activation_df)
                and activation_df.columns.equals(self.default_activation_df.columns)
            ):
                rows_to_delete = []
                modified_activation_df = deepcopy(activation_df)
                for (colname, colval) in activation_df.iteritems():
                    if self.default_df_descriptor[colname][0] == 'string':
                        # if 'string' type is defined in default_df_descriptor, then
                        # convert values into string
                        if not all(activation_df[colname].isnull()):
                            modified_activation_df[colname] = colval.map(str)
                        # check if possibles values are defined from
                        # architecture_df
                        if not all(self.default_activation_df[colname].isnull()):
                            # check if values are among possible values
                            if not all(
                                colval.isin(
                                    self.default_activation_df[colname])
                            ):
                                wrong_values = colval.loc[
                                    ~colval.isin(
                                        self.default_activation_df[colname])
                                ]
                                rows_to_delete.extend(
                                    wrong_values.index.values.tolist()
                                )
                                self.ee.logger.error(
                                    f'Invalid Value Block Activation Configuration: {wrong_values.values.tolist()} in column {colname} not in *possible values* {self.default_activation_df[colname].values.tolist()}'
                                )

                    elif self.default_df_descriptor[colname][0] == 'bool':
                        # if 'bool' type is defined in default_df_descriptor, then
                        # check if value block is available
                        if not all(self.default_activation_df[colname]):
                            for vb in self.activation_columns:
                                unavailable_vb = self.default_activation_df.loc[
                                    ~self.default_activation_df[colname], vb
                                ].values.tolist()
                                if not (
                                    activation_df.loc[
                                        activation_df[vb].isin(unavailable_vb)
                                    ][activation_df[colname]]
                                ).empty:
                                    # if not available value blocks are activated,
                                    # set False in activation_df
                                    self.ee.logger.error(
                                        f'Invalid Value Block Activation Configuration: value block {colname} not available for {list(set(activation_df.loc[activation_df[vb].isin(unavailable_vb)][activation_df[colname]][vb].values.tolist()))}'
                                    )
                                    modified_activation_df.loc[
                                        modified_activation_df[vb].isin(
                                            unavailable_vb), colname
                                    ] = False
                        # if colname value block is desactivated, then
                        # desactivate its children
                        if False in activation_df[colname].values.tolist():
                            children_names = self.get_children_names(
                                colname, self.architecture_df
                            )
                            if len(children_names) > 0:
                                modified_activation_df.loc[
                                    ~modified_activation_df[colname], children_names
                                ] = False

                if len(rows_to_delete) > 0:
                    # remove rows with values not among possible_values
                    self.get_data_in()[self.ACTIVATION_DF][self.VALUE] = modified_activation_df.drop(
                        rows_to_delete
                    )

    def get_children_names(self, parent_name, architecture):
        """
        Recursive method to get children names for parent name by reading architecture_df
        """
        if parent_name in architecture[self.PARENT].values.tolist():
            return architecture.loc[
                architecture[self.PARENT] == parent_name, self.CURRENT
            ].values.tolist()
        else:
            for sub_architecture in [
                action[3]
                for action in architecture[self.ACTION]
                if action[0] == 'scatter_architecture'
            ]:
                return self.get_children_names(parent_name, sub_architecture)
        return []

    def create_vb_disc_namespaces_and_builders(
        self, archi_df, activation_dict, archi_parent
    ):
        """
        Create builder dict of value blocks
        """
        builder_dict = {}

        for index, row in archi_df.iterrows():
            if row[self.PARENT] is not None:
                # if current element has parent
                namespace = f'{row[self.PARENT]}.{row[self.CURRENT]}'
            else:
                # if current element is at first node of architecture
                namespace = row[self.CURRENT]

            namespace_list, activation_dict = self.get_full_namespaces_from_archi(
                namespace, activation_dict, archi_df, archi_parent
            )

            for ns in namespace_list:
                if ns.startswith(f'{archi_parent}.'):
                    # split namespace to get builder_name without architecture
                    # name
                    builder_name = '.'.join(ns.split('.')[1:])
                else:
                    # builder_name is ns if ns does not contain architecture name
                    # then the discipline will be built just below architecture
                    builder_name = ns

                # get the builder from factory even if row[self.TYPE] is a
                # tuple (folder,class_name)
                disc_builder = self.get_builder_from_factory(
                    builder_name, row[self.BUILDER_TYPE])

                if ns in builder_dict:
                    builder_dict[ns].append(disc_builder)
                else:
                    builder_dict[ns] = [disc_builder]

        return builder_dict, activation_dict

    def get_builder_from_factory(self, builder_name, builder_def):
        '''
        Get a builder from the ee factory, two ways of working : 
            - builder_def is a string : find the builder_def class in each folder of the folder_list
            - builder_def is a tuple : find builder_def[1] class name in the folder given as first element of the tuple builder_def[0] 
        '''
        if isinstance(builder_def, tuple):
            disc_builder = self.ee.factory.get_builder_from_class_name(
                builder_name, builder_def[1], [builder_def[0]]
            )
        else:

            disc_builder = self.ee.factory.get_builder_from_class_name(
                builder_name, builder_def, self.vb_folder_list
            )

        return disc_builder

    def get_children_list_by_vb(self, builder_dict):
        """
        Get direct children (not grand children) to gather outputs of your children when you are a father
        and to know if a father must be build if he has no children
        """
        for ns in builder_dict:
            self.children_dict[ns] = []
            for ns_child in builder_dict:
                if (
                    ns in ns_child
                    and ns != ns_child
                    and len(ns.split('.')) + 1 == len(ns_child.split('.'))
                ):
                    self.children_dict[ns].append(ns_child)

    def get_full_namespaces_from_archi(
        self, namespace, activation_dict, archi_df, archi_parent
    ):
        """
        Get full namespaces of builder with current namespace by reading archi_df
        """
        new_namespace_list = []

        if namespace == archi_parent:
            # if namespace is archi_parent name, we want to build value
            # block at first node of architecture
            new_namespace_list.append(namespace)
        else:
            parent_name = archi_parent.split('.')[-1]
            # if namaspace starts with parent_name,
            # namespace does not have parents
            if namespace.startswith(f'{parent_name}.'):
                namespace_wo_parent = namespace.replace(f'{parent_name}.', '')
                full_namespace = f'{archi_parent}.{namespace_wo_parent}'
                new_namespace_list.append(full_namespace)
            else:
                # get parents of namespace
                vb_father = archi_df.loc[
                    (archi_df[self.CURRENT] == namespace.split('.')[0])
                    & (~archi_df[self.PARENT].isna())
                ]
                # if no parents and architecture name not in architecture_df,
                # namespace builder will be built below architecture
                if len(vb_father) == 0:
                    new_namespace_list.append(namespace)

                # get list of namespaces created with list of parents
                for i, vb in vb_father.iterrows():
                    namespace_with_father = f'{vb[self.PARENT]}.{namespace}'

                    (
                        ns_list_father,
                        activation_dict,
                    ) = self.get_full_namespaces_from_archi(
                        namespace_with_father, activation_dict, archi_df, archi_parent
                    )

                    if vb[self.PARENT] in activation_dict.keys():
                        activation_dict[vb[self.PARENT]].update(
                            {namespace_with_father: vb[self.CURRENT]}
                        )

                    new_namespace_list.extend(ns_list_father)

        return new_namespace_list, activation_dict

    def get_action_builder(self, namespace, archi_df):
        """
        Get action and args of builder_name from architecture_df
        """
        if '.' not in namespace:
            # get action of namespace without parent
            action = archi_df[archi_df[self.CURRENT] == namespace][self.ACTION].values[
                0
            ]
        else:
            # get action of namespace splitted into current/parent
            parent_name, current_name = namespace.split('.')[-2:]
            action = archi_df.loc[
                (archi_df[self.CURRENT] == current_name)
                & (archi_df[self.PARENT] == parent_name),
                self.ACTION,
            ].values[0]
        if isinstance(action, (str)):
            return action, ()
        elif isinstance(action, (tuple)):
            return action[0], action[1:]
        else:
            raise ArchiBuilderException(
                f'Invalid Action in architecture dataframe. Action must be among: {list(self.POSSIBLE_ACTIONS.values())}'
            )

    def build(self):
        """
        Build method to build all value blocks regarding the architecture
        """

        self.check_activation_df()

        activ_builder_dict, self.builder_dict = self.build_action_from_builder_dict(
            self.builder_dict, self.architecture_df
        )

        # build activated builders
        for namespace in self.builder_dict.keys():
            if namespace in activ_builder_dict:

                # len(activ_builder_dict[namespace]) == 1 or
                # isinstance(self.builder_dict[namespace], list):
                for builder in activ_builder_dict[namespace]:
                    disc = self.build_value_block(builder)
                    if builder.sos_name.endswith('@archi_node'):
                        # builder at archi_node should have the same display than architecture builder
                        local_ns = self.ee.ns_manager.get_local_namespace(disc)
                        local_ns.set_display_value(self.get_disc_display_name())
                    if namespace not in self.archi_disciplines:
                        self.ee.factory.add_discipline(disc)
                        self.archi_disciplines[namespace] = [disc]
                    elif disc not in self.archi_disciplines[namespace]:
                        self.ee.factory.add_discipline(disc)
                        self.archi_disciplines[namespace].append(disc)

            # clean builders not activated anymore
            elif namespace in self.archi_disciplines:
                self.clean_children(self.archi_disciplines[namespace])
                self._is_configured = False
                del self.archi_disciplines[namespace]

        self.activated_builders = activ_builder_dict

        self.send_children_to_father()

    def build_value_block(self, builder):
        """
        Method to build discipline with builder and namespace
        """
        # build discipline below architecture
        self.associate_namespace_to_sub_builder(builder)
        discipline = builder.build()

        return discipline

    def build_action_from_builder_dict(self, builder_dict, archi_df):
        """
        Recursive method to get builder_dict and activ_builder_dict by reading archi_df
        """
        activ_builder_dict = {}
        new_builder_dict_list = []
        new_activ_builder_dict_list = []

        for namespace, builder_list in builder_dict.items():
            for builder in builder_list:
                if '.' in namespace:
                    builder_name = namespace.split('.')[-1]
                else:
                    # builder name at architecture node
                    builder_name = namespace
                    if not builder.sos_name.endswith('@archi_node'):
                        old_builder_name = builder.sos_name
                        builder.set_disc_name(f'{builder.sos_name}@archi_node')
                        # self.ee.ns_manager.add_display_ns_to_builder(builder, old_builder_name)
                action, args = self.get_action_builder(namespace, archi_df)

                if self.is_builder_activated(namespace, builder_name):
                    if namespace in activ_builder_dict:
                        activ_builder_dict[namespace].append(builder)
                    else:
                        activ_builder_dict[namespace] = [builder]

                    if f'{builder}@{action}' in self.special_builders:
                        activ_builder_dict[namespace].extend(self.special_builders[f'{builder}@{action}'])

                    else:

                        if action == 'architecture':
                            # First add the current builder to the activated builder
                            # if self.is_builder_activated(namespace,
                            # builder_name):

                            # Check the arguments of the action architecture
                            if not isinstance(args[0], pd.DataFrame):
                                raise ArchiBuilderException(
                                    f'Action for builder \'{builder_name}\' must be defined by a tuple: {self.POSSIBLE_ACTIONS[action[0]]}, with a dataframe \'architecture_df\''
                                )
                            # build with this architecture
                            new_builder_dict, new_activation_dict = self.get_subarchi_builders(
                                args[0], namespace
                            )

                            self.activation_dict.update(new_activation_dict)

                            (
                                new_activ_builder_dict,
                                new_builder_dict,
                            ) = self.build_action_from_builder_dict(new_builder_dict, args[0])

                            # Store new infos in global variables
                            # Check if the new architecture is already in
                            # self.architecture_df (by checking isin and size of
                            # df)

                            if args[0][['Parent', 'Current']].isin(
                                    self.architecture_df[['Parent', 'Current']]).to_numpy().sum() != np.prod(
                                list(args[0][['Parent', 'Current']].shape)
                            ):
                                self.architecture_df = pd.concat(
                                    [self.architecture_df, args[0]], ignore_index=True
                                )

                            # store new builders of sub architecture
                            new_builder_dict_list.append(new_builder_dict)
                            # store new activates builders of sub architecture
                            new_activ_builder_dict_list.append(
                                new_activ_builder_dict)

                        elif action == 'scatter':

                            if isinstance(args[1], tuple):
                                # get builder of scatter of scatter
                                raise Exception('Nested scatters in architecture builder not implemented')
                            else:
                                # get builder of scatter
                                scatter_builder = self.get_builder_from_factory(
                                    namespace, args[1])
                                activ_builders = self.build_action_scatter(
                                    namespace, args[0], scatter_builder
                                )

                            self.special_builders[f'{builder}@{action}'] = activ_builders
                            # add builders of scatter if not already in
                            # activ_builder_dict
                            for scatter_activ_builder in activ_builders:
                                if scatter_activ_builder not in activ_builder_dict[namespace]:
                                    activ_builder_dict[namespace].append(
                                        scatter_activ_builder)

                        elif action == 'scatter_architecture':
                            if not isinstance(
                                    args[2], pd.DataFrame
                            ):
                                raise ArchiBuilderException(
                                    f'Action for builder \'{builder_name}\' must be defined by a tuple: {self.POSSIBLE_ACTIONS[action[0]]}, with a string \'var_name\', a string \'builder_class\' and a dataframe \'architecture_df\''
                                )
                            scatter_list_name = args[0]
                            # get initial builders_dict and activation_dict for
                            # sub architecture
                            new_builder_dict, new_activation_dict = self.get_subarchi_builders(
                                args[2], namespace
                            )

                            self.activation_dict.update(new_activation_dict)

                            (
                                new_activ_builder_dict,
                                new_builder_dict,
                            ) = self.build_action_from_builder_dict(new_builder_dict, args[2])

                            archi_builder_list = [
                                bd
                                for sublist in list(new_activ_builder_dict.values())
                                for bd in sublist
                            ]

                            # Need to modify names of builder
                            for builder_list in archi_builder_list:
                                builder_list.set_disc_name(
                                    builder_list.sos_name.split(
                                        f'{builder_name}.')[-1]
                                )

                            # get builders of scatter_architecture
                            # possibility to instantiate several builder at scatter node
                            # builders are stored in a list even if there is only
                            # one
                            if isinstance(args[1], list):
                                scatter_builder_cls_list = [self.get_builder_from_factory(
                                    namespace, builder_def) for builder_def in args[1]]

                            else:
                                scatter_builder_cls_list = [self.get_builder_from_factory(
                                    namespace, args[1])]

                            # we send a  list of scatter_builder_cls_list to
                            # prevent from multiple builders at scatter node
                            activ_builders = self.get_scatter_builder(
                                namespace,
                                scatter_list_name,
                                archi_builder_list
                            )

                            # add scatter_architecture builders in
                            # activ_builder_dict
                            self.special_builders[f'{builder}@{action}'] = activ_builders
                            activ_builder_dict[namespace].extend(activ_builders)

        # add builders of sub architectures in builder_dict
        for new_builder_dict in new_builder_dict_list:
            for new_ns, builder_list in new_builder_dict.items():
                if new_ns not in builder_dict:
                    builder_dict[new_ns] = builder_list
            # add builders in new_builder_dict in architecture children
            self.get_children_list_by_vb(new_builder_dict)

        # add activated builders of sub architectures in activ_builder_dict
        for new_activ_builder_dict in new_activ_builder_dict_list:
            for new_ns, builder_list in new_activ_builder_dict.items():
                if new_ns not in activ_builder_dict:
                    activ_builder_dict[new_ns] = builder_list
                else:
                    for bd in builder_list:
                        if bd.cls not in [
                            abd.cls for abd in activ_builder_dict[new_ns]
                        ]:
                            activ_builder_dict[new_ns].append(bd)

        return activ_builder_dict, builder_dict

    def build_action_scatter(
            self,
            namespace,
            scatter_list_name,
            scatter_builder,
    ):
        """
        Build a scatter under a node
         namespace : namespace of the node where to build the scatter
         scatter_map_name : name ot he scatter_map associatd to the scatter
         sub_scatter_builder : in case of scatter of scatter the builder to scatter at the last level
         builder_on_first_scatter : in case of scatter of scatter, the builder to build at the place of the scattered_name
         ex : a sumbuilder at the AC node

        """

        activ_builders = self.get_scatter_builder(
            namespace,
            scatter_list_name,
            scatter_builder
        )

        return activ_builders

    def is_builder_activated(self, namespace, builder_name):
        """
        Return True/False if builder is activated/desactivated in self.activation_df
        """
        if (
            self.ACTIVATION_DF in self.get_data_in()
            and self.get_sosdisc_inputs(self.ACTIVATION_DF) is not None
        ):
            activation_df = self.get_sosdisc_inputs(self.ACTIVATION_DF)
            if (
                builder_name in activation_df.columns
                and builder_name not in self.activation_dict.keys()
            ):
                df = deepcopy(activation_df)
                for builder, activ_dict in self.activation_dict.items():
                    if namespace in activ_dict:
                        df = df.loc[df[builder] == activ_dict[namespace]]
                return True in df[builder_name].values
            else:
                return True
        else:
            return False

    def send_children_to_father(self):
        """
        Send the list of children (direct disc object) to the father discipline in the
        attribute children_list
        """
        for ns, disc_list in self.archi_disciplines.items():
            # scatter_in_node = False
            # scattered_disciplines = {}
            for disc in disc_list:
                disc_children_list = [
                    child
                    for ns_child in self.children_dict[ns]
                    if ns_child in self.archi_disciplines
                    for child in self.archi_disciplines[ns_child]
                    if not hasattr(child, 'LINKED_MODE')
                ]
                disc.add_disc_list_to_config_dependency_disciplines(
                    disc_children_list)

    def setup_sos_disciplines(self):
        """
        Set scenario_df value by reading activation_df input
        """
        dynamic_outputs = {}
        for driver_name, input_name in self.driver_input_to_fill.items():

            if f'{driver_name}.scenario_df' in self.get_data_out():
                activation_df = deepcopy(self.get_sosdisc_inputs(self.ACTIVATION_DF))

                if driver_name == 'driver':
                    # means that we have a driver at archi node
                    namespace = self.sos_name
                    builder_name = self.sos_name
                else:
                    driver_display_name = driver_name.replace('.driver', '')
                    namespace = f"{self.sos_name}.{driver_display_name}"
                    builder_name = driver_display_name.split('.')[-1]

                for var, activ_dict in self.activation_dict.items():
                    if namespace in activ_dict:
                        activation_df = activation_df.loc[
                            activation_df[var] == activ_dict[namespace]]
                        break
                try:
                    subactivation_df = activation_df.loc[activation_df[builder_name]]
                except:
                    pass
                input_value = [
                    val for val in subactivation_df[input_name] if val is not None
                ]
                indexes = np.unique(input_value, return_index=True)[1]
                input_value = [input_value[index] for index in sorted(indexes)]

                scenario_full_name = self.get_var_full_name(f'{driver_name}.scenario_df', self.get_data_out())
                self.dm.set_data(scenario_full_name, 'value', pd.DataFrame({self.SCENARIO_NAME: input_value,
                                                                            self.SELECTED_SCENARIO: True}),
                                 check_value=False)
            dynamic_outputs.update(
                {f'{driver_name}.scenario_df': {'type': 'dataframe',
                                                'default': pd.DataFrame(
                                                    columns=(self.SCENARIO_NAME, self.SELECTED_SCENARIO))},
                 f'{driver_name}.builder_mode': {'type': 'string',
                                                 'value': 'multi_instance'}},
            )

        self.add_outputs(dynamic_outputs)

    def get_scatter_builder(
            self,
            namespace,
            scatter_list_name,
            builder,
    ):
        """
        Get builders list for scatter action at namespace node
        """

        result_builder_list = []
        # case when scatter is on archi_node or scatter on other node at root level
        if namespace == self.sos_name:
            driver_name = 'driver'
        # standard case
        elif self.sos_name in namespace:
            builder_name = f"{namespace.split('.', 1)[1]}"
            driver_name = f'{builder_name}.driver'
            # case when no archi_node is specified in architecture_df
        else:
            builder_name = namespace
            driver_name = f'{builder_name}.driver'

        if isinstance(builder, list):

            builder_scatter = self.ee.factory.create_driver(driver_name, builder)
        else:
            builder.set_disc_name(builder.sos_name.split('.')[-1])
            builder_scatter = self.ee.factory.create_driver(driver_name, [builder],
                                                            display_options={'hide_under_coupling': True},
                                                            flatten_subprocess=True)
        if namespace == self.sos_name:
            self.ee.ns_manager.add_display_ns_to_builder(
                builder_scatter[0], self.get_disc_full_name())
        elif len(namespace.split('.')) == 1:
            self.ee.ns_manager.add_display_ns_to_builder(
                builder_scatter[0], self.get_disc_full_name().replace(self.sos_name, namespace))
        else:
            # builder_name = f"{namespace.split('.', 1)[1]}"
            self.ee.ns_manager.add_display_ns_to_builder(
                builder_scatter[0], f'{self.get_disc_full_name()}.{builder_name}')
        result_builder_list.extend(builder_scatter)

        input_name = scatter_list_name
        # set input_name value in data_in by reading activated children in
        # activation_df input
        self.driver_input_to_fill[driver_name] = input_name

        return result_builder_list

    def set_scatter_list_under_scatter(self, full_input_name, input_value):
        """
        Function to set the scatter_list und er the corresponding scatter to create children
        """

        if full_input_name in self.ee.dm.data_id_map:
            self.ee.dm.set_data(full_input_name, self.EDITABLE, False)
            self.ee.dm.set_data(full_input_name, self.VALUE, input_value)

    def get_subarchi_builders(self, subarchi_df, parent_namespace):
        """
        Build initial builder_dict and activation_dict by reading subarchi_df
        """
        sub_builder_dict, sub_activation_dict = self.builder_dict_from_architecture(
            subarchi_df, parent_namespace
        )

        return sub_builder_dict, sub_activation_dict

    def delete_father_without_children(self, activate_dict):
        """
        Do not build a father which does not have children
        """
        new_children_dict = {}
        children_dict_size_old = len(self.children_dict)
        while children_dict_size_old != len(self.children_dict):
            children_dict_size_old = len(new_children_dict)
            new_children_dict = {}
            for namespace, children_list in self.children_dict.items():
                if len(children_list) != 0:
                    new_children_dict[namespace] = children_list

        return {
            namespace: builder
            for namespace, builder in activate_dict.items()
            if namespace in new_children_dict
        }

    def create_activation_df(self, archi_df):
        """
        Create activation_df with all value blocks activated for all actor by default
        """
        activation_dict = {}
        df_dict = {}
        df_descriptor = {}

        # add activation columns , the archibuider shouldn't be an activation
        # row
        activation_df_row = archi_df.loc[
            (archi_df[self.ACTIVATION]) & (
                archi_df[self.CURRENT] != self.sos_name)
        ][self.CURRENT].values
        for builder in activation_df_row:
            parent = archi_df.loc[archi_df[self.CURRENT] == builder][
                self.PARENT
            ].values[0]
            activation_dict[parent] = {}
            if parent not in df_dict:
                self.activation_columns.append(parent)
                df_descriptor[parent] = ('string', None, True)
                df_dict[parent] = [builder]
            else:
                df_dict[parent].append(builder)

        # add scatter inputs columns
        scatter_columns = [
            action[1]
            for action in archi_df[self.ACTION]
            if action[0] in ['scatter', 'scatter_architecture']
        ]
        scatter_of_scatter_columns = []
        for action in archi_df[self.ACTION]:
            if action[0] == 'scatter' and isinstance(action[2], tuple):
                scatter_of_scatter_columns.append(action[2][1])

        for input_name in scatter_columns + scatter_of_scatter_columns:
            df_descriptor[input_name] = ('string', None, True)
            df_dict[input_name] = None

        # add other columns
        activation_df_column = [
            builder
            for builder in archi_df[self.CURRENT].values
            if builder not in activation_df_row
        ]

        # we check if there's a node defined as root_node in the architecture data frame
        # We consider we are facing a root_node situation if the architecture name appears
        # both in current and parent columns in the archi_df (cf
        # build_architecture method)
        root_node = False
        current_list = list(archi_df[self.CURRENT])
        parent_list = list(archi_df[self.PARENT])
        if (self.sos_name in current_list) and (self.sos_name in parent_list):
            root_node = True
        # In case of a root_node, we redefine the activation_column so that the
        # root_node is not taken into account
        if root_node:
            activation_df_column = [
                builder
                for builder in archi_df[self.CURRENT].values
                if ((builder not in activation_df_row) and (builder != self.sos_name))
            ]
        for builder in activation_df_column:
            df_descriptor[builder] = ('bool', None, True)
            df_dict[builder] = True
            if len(activation_df_row) > 0:
                # check in architecture_df if value block is available
                parent_list = (
                    archi_df[self.PARENT]
                    .loc[archi_df[self.CURRENT] == builder]
                    .values.tolist()
                )
                if len(parent_list) == 1 and parent_list[0] in activation_df_column:
                    df_dict[builder] = df_dict[parent_list[0]]
                else:
                    df_dict[builder] = [
                        activ_row in parent_list for activ_row in activation_df_row
                    ]

        # create activation_df
        try:
            activation_df = pd.DataFrame(df_dict)
        except:
            try:
                activation_df = pd.DataFrame(
                    df_dict, index=range(len(self.default_activation_df))
                )
            except:
                activation_df = pd.DataFrame(df_dict, index=[0])

        if self.default_activation_df is None:
            self.default_activation_df = activation_df
        else:
            for column in activation_df:
                if column not in self.default_activation_df:
                    self.default_activation_df[column] = activation_df[column]

        if self.ACTIVATION_DF not in self.get_data_in():
            dict_desc_in = {
                self.ACTIVATION_DF: {
                    ProxyDisciplineBuilder.TYPE: 'dataframe',
                    ProxyDisciplineBuilder.EDITABLE: True,
                    ProxyDisciplineBuilder.DEFAULT: self.default_activation_df,
                    ProxyDisciplineBuilder.DATAFRAME_DESCRIPTOR: df_descriptor,
                    ProxyDisciplineBuilder.DATAFRAME_EDITION_LOCKED: False,
                    ProxyDisciplineBuilder.STRUCTURING: True,
                }
            }
            self.default_df_descriptor = df_descriptor
            self.add_inputs(dict_desc_in, clean_inputs=False)
        else:
            if not self.default_activation_df.equals(
                self.dm.get_data(
                    self.get_var_full_name(
                        self.ACTIVATION_DF, self.get_data_in()),
                    ProxyDisciplineBuilder.DEFAULT,
                )
            ):
                self.get_data_in()[self.ACTIVATION_DF][
                    self.VALUE
                ] = self.default_activation_df
                self.get_data_in()[self.ACTIVATION_DF][
                    self.DEFAULT
                ] = self.default_activation_df
            self.default_df_descriptor.update(df_descriptor)
            self.get_data_in()[self.ACTIVATION_DF][self.DATAFRAME_DESCRIPTOR].update(
                df_descriptor
            )

        return activation_dict

    def get_scatter_input_value(
        self, namespace, input_name, builder_name, condition_dict=None
    ):
        """
        Get product list of actor_name for builder_name
        """
        if self.ACTIVATION_DF in self.get_data_in():
            activation_df = deepcopy(
                self.get_sosdisc_inputs(self.ACTIVATION_DF))
            for var, activ_dict in self.activation_dict.items():
                if namespace in activ_dict:
                    activation_df = activation_df.loc[
                        activation_df[var] == activ_dict[namespace]
                    ]

            subactivation_df = activation_df.loc[activation_df[builder_name]]
            # To deal with scatter of scatter
            # Condition_dict will look like : {'AC_list':'AC1'}
            if condition_dict is not None:
                for key, value in condition_dict.items():
                    subactivation_df = subactivation_df.loc[
                        subactivation_df[key] == value
                    ]

            input_value = [
                val for val in subactivation_df[input_name] if val is not None
            ]
            indexes = np.unique(input_value, return_index=True)[1]
            input_value = [input_value[index] for index in sorted(indexes)]

        return input_value

    def is_configured(self):
        return ProxyDiscipline.is_configured(self) and all(
            [input in self.get_data_in().keys()
             for input in self.inst_desc_in.keys()]
        )

    def remove_discipline_list(self, disc_list):
        """remove one discipline from coupling"""
        for disc in disc_list:
            #             if isinstance(disc, SoSDisciplineScatter):
            #                 disc.clean_scattered_disciplines([])
            self.ee.root_process.sos_disciplines.remove(disc)
            disc.clean_dm_from_disc()
            self.ee.ns_manager.remove_dependencies_after_disc_deletion(
                disc, self.disc_id
            )

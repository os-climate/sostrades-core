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
from sostrades_core.tools.tree.data_management_discipline import DataManagementDiscipline
from sostrades_core.execution_engine import ns_manager
"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
"""
from sostrades_core.tools.tree.treenode import TreeNode
from sostrades_core.execution_engine.ns_manager import NamespaceManager, NS_SEP
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline

IO_TYPE = ProxyDiscipline.IO_TYPE
TYPE_IN = ProxyDiscipline.IO_TYPE_IN
TYPE_OUT = ProxyDiscipline.IO_TYPE_OUT
VISI = ProxyDiscipline.VISIBILITY


class TreeView:
    """
    Tree view class
    """
    PROCESS_DOCUMENTATION = 'Process documentation'

    def __init__(self, name, no_data=False, read_only=False, exec_display=False):
        """ class constructor
        """
        self.name = name
        self.no_data = no_data
        self.read_only = read_only
        self.root = None
        self.exec_display = exec_display

    def create_tree_node(self, data_manager, root_process, ns_manager, process_module=''):
        """ Function that builds a composite structure (tree view  of tree nodes)
        regarding the DataManager stored through disciplines references and data dictionary

        :params: data_manager, execution engine data manager
        :type: DataManager

        :params: root process, main discipline (root discipline)
        :type: ProxyDiscipline

        :params: ns_manager, namespace manager use to store variable defined outside a discipline
        :type: NamespaceManager

        :params: process_module, module name use to generated the process
        :type: string
        """

        disc_dict = data_manager.disciplines_dict

        treenodes = {}

        # Initialise treeview root discipline on treeview

        treenode = self.add_treenode(root_process)
        treenodes[self.root.full_namespace] = self.root

        documentation_folder = ''
        try:
            from importlib import import_module
            documentation_folder = import_module(process_module).__file__

            if documentation_folder != '':

                self.root.add_markdown_documentation(TreeNode.get_markdown_documentation(
                    documentation_folder), TreeView.PROCESS_DOCUMENTATION)
        except:
            pass

        # First create the tree structure regarding the hosted process
        # Getting the key of the discipline dictionary and order them allow to have
        # the full view of the tree (keeping the discipline with dotted name
        # Correct treenode built need to have nemaspace ordered in alphebetical
        # order

        self.create_treenode_rec(self.root, treenodes, disc_dict)

        data_dict = data_manager.convert_data_dict_with_display_name(
            self.exec_display)

        # Now populate each node with their process parameter
        for key, val in data_dict.items():
            display_key = val['display_name']
            # Each key contains variable namespace followed by its name
            # ns1.ns1.variable_name
            # Special case for public variables which are associated to the upper node level
            # (without namespace name)
            namespace = NamespaceManager.compose_ns(
                display_key.split(NS_SEP)[:-1])

            if namespace in treenodes:
                treenode = treenodes[namespace]
                self.set_treenode_data(treenode, key, val, disc_dict)
                self.set_treenode_discipline_data(treenode, key, val, disc_dict)

            else:
                try:  # Todo review this code because access on exec engine attribute is not correct
                        # Also do not forget this is here to hide misplaced
                        # output variables in treeview (ns_ac related)
                    if val['io_type'] == 'in':
                        treenode = self.add_treenode(
                            None, namespace.split(NS_SEP))
                        self.set_treenode_data(treenode, key, val, disc_dict)
                        self.set_treenode_discipline_data(treenode, key, val, disc_dict)
                except:
                    pass

        # Add a tree node if the post processing namespace does not exist in
        # the treenodes
        for namespace in ns_manager.ee.post_processing_manager.namespace_post_processing:
            try:
                ns_list = ns_manager.get_all_namespace_with_name(namespace).get_value(
                )
                for ns in ns_list:
                    ns_value = ns.get_value()
                    if ns_value not in treenodes.keys():
                        treenode = self.add_treenode(
                            None, ns_value.split(NS_SEP))
                        treenode.full_namespace = ns_value
            except:
                pass

    def set_treenode_data(self, treenode, key, val, disc_dict):

        if not self.no_data:

            treenode.data[key] = {k: v for k, v in val.items()}

            # retrieve model name full path for variable key
            model_name_full_path = val['model_origin']
            io_type = val['io_type']
            if val['model_origin'] in disc_dict.keys():
                discipline_info = disc_dict[val['model_origin']]
                model_name_full_path = discipline_info["model_name_full_path"]
                var_name = val['var_name']
                # Check if the data is a stron,g coupling (the ee set it to input but it is an output)
                if val['io_type'] == 'in' and var_name not in discipline_info['reference']._data_in.keys() \
                    and var_name in discipline_info['reference']._data_out.keys():
                    io_type = 'out'

            treenode.data[key][ProxyDiscipline.VARIABLE_KEY] = treenode.create_data_key(model_name_full_path, io_type, val['var_name'])

            if key in treenode.disc_data:
                treenode.data[key][ProxyDiscipline.DISCIPLINES_FULL_PATH_LIST] = \
                    treenode.disc_data[key][ProxyDiscipline.DISCIPLINES_FULL_PATH_LIST]

            if self.read_only:
                treenode.data[key][ProxyDiscipline.EDITABLE] = False

    def set_treenode_discipline_data(self, treenode, key, val, disc_dict):

        if not self.no_data:
            temp_data = {k: v for k, v in val.items()}

            
            

            # retrieve model name full path for variable key
            model_name_full_path = val['model_origin']
            io_type = val['io_type']
            if val['model_origin'] in disc_dict.keys():
                discipline_info = disc_dict[val['model_origin']]
                model_name_full_path = discipline_info["model_name_full_path"]
                var_name = val['var_name']
                # Check if the data is a stron,g coupling (the ee set it to input but it is an output)
                if val['io_type'] == 'in' and var_name not in discipline_info['reference']._data_in.keys() \
                    and var_name in discipline_info['reference']._data_out.keys():
                    io_type = 'out'

            temp_data[ProxyDiscipline.VARIABLE_KEY] = treenode.create_data_key(model_name_full_path, io_type, val['var_name'])

            if self.read_only:
                temp_data[ProxyDiscipline.EDITABLE] = False
            

            if key not in treenode.disciplines_by_variable.keys():
                # create data management discipline DATA
                data_manamement_data_key = 'Data'
                if not data_manamement_data_key in treenode.data_management_disciplines:
                    data_management_discipline = DataManagementDiscipline()
                    data_management_discipline.namespace = treenode.full_namespace
                    data_management_discipline.model_name_full_path = "Data"
                    data_management_discipline.discipline_label = "Data"
                    treenode.data_management_disciplines[data_manamement_data_key] = data_management_discipline

                if temp_data[ProxyDiscipline.NUMERICAL]:
                    treenode.data_management_disciplines[data_manamement_data_key].numerical_parameters[key] = temp_data
                elif temp_data[ProxyDiscipline.IO_TYPE] == 'in':
                    treenode.data_management_disciplines[data_manamement_data_key].disciplinary_inputs[key] = temp_data
                elif temp_data[ProxyDiscipline.IO_TYPE] == 'out':
                    treenode.data_management_disciplines[data_manamement_data_key].disciplinary_outputs[key] = temp_data
                else:
                    temp_data[ProxyDiscipline.DISCIPLINES_FULL_PATH_LIST] = treenode.disciplines_by_variable[key]

            else:
                for discipline_key in treenode.disciplines_by_variable[key]:
                    if temp_data[ProxyDiscipline.NUMERICAL]:
                        treenode.data_management_disciplines[discipline_key].numerical_parameters[key] = temp_data
                    elif temp_data[ProxyDiscipline.IO_TYPE] == 'in':
                        treenode.data_management_disciplines[discipline_key].disciplinary_inputs[key] = temp_data
                    elif temp_data[ProxyDiscipline.IO_TYPE] == 'out':
                        treenode.data_management_disciplines[discipline_key].disciplinary_outputs[key] = temp_data



    def add_treenode(self, discipline, namespace=None):
        """ Add a new treenode to the treeview.
        Treenode position is driven using discipline attribute from the root node

        :params: discipline, discipline node to add
        :type: ProxyDiscipline

        :params: children_namespace, clidren namespace to navigate
        :type: string[]

        :return: TreeNode
        """

        if namespace is None:
            namespace = discipline.get_disc_display_name(
                self.exec_display).split(NS_SEP)

        if len(namespace) > 0:

            if self.root is None:
                self.root = TreeNode(namespace[0])

            children_namespace = []
            if len(namespace) > 1:
                children_namespace = namespace[1:]

            return self.__add_treenode(self.root, discipline, children_namespace)

    def __add_treenode(self, current_treenode, discipline, children_namespace):
        """ Recursively look into each treenode to find/add a new treeenode regarding
        the namespace place
        Return the treenode created/find at the end of the namespace

        :params: current_treenode, current treenode search pointer
        :type: TreeNode

        :params: discipline, discipline node to add
        :type: ProxyDiscipline

        :params: children_namespace, clidren namespace to navigate
        :type: string[]

        :return: TreeNode
        """

        if current_treenode is None:
            pass  # raise error

        if len(children_namespace) > 0:

            children_to_search = children_namespace[0]

            # look for children with first namespace name
            tree_node = next(
                (tn for tn in current_treenode.children if tn.name == children_to_search), None)

            if tree_node is None:
                # no child so create a new treenode
                tree_node = TreeNode(children_to_search)
                current_treenode.children.append(tree_node)

            if len(children_namespace) > 1:
                return self.__add_treenode(tree_node, discipline, children_namespace[1:])
            else:
                return self.__add_treenode(tree_node, discipline, [])

        else:
            if discipline is not None:
                current_treenode.update_treenode_attributes(
                    discipline, self.no_data, self.read_only)
            return current_treenode

    def create_treenode_rec(self, current_treenode, treenodes, disc_dict):
        """ Recursive method that create treenode structure regarding
        the SoSDisci
        """

        # Retireve current tree node discipline
        if current_treenode.identifier in disc_dict:
            # Retrieve the list of sub disciplines to map as Treenode
            current_discipline = disc_dict[current_treenode.identifier]['reference']

            if current_discipline is not None:
                sub_disciplines = current_discipline.ordered_disc_list

                for sub_discipline in sub_disciplines:
                    new_treenode = self.add_treenode(sub_discipline)
                    treenodes[new_treenode.full_namespace] = new_treenode

                    self.create_treenode_rec(
                        new_treenode, treenodes, disc_dict)

    def to_json(self):
        return self.root.to_json()

    def to_dict(self):
        return self.root.to_dict()

    def __str__(self):
        return str(self.root)

    def display_nodes(self, display_variables=None):

        def display_node(node, level=0,
                         display_variables=display_variables):
            str_nodes = '\n' + '\t' * level + '|_ ' + node.name
            level += 1
            # be able to display variables at their node level grouped by
            # in/out
            if display_variables is not None:
                tabs = '\t' * level

                def get_variables_info(str_nodes, node_data, io_type, disp_detail='var_name'):
                    if io_type == TYPE_IN:
                        io_t = '->'
                    else:
                        io_t = '<-'
                    for var, v_d in sorted(node_data.items()):
                        if v_d[IO_TYPE] == io_type:
                            str_nodes += f'\n{tabs}{io_t} '
                            if 'visi' in str(disp_detail).lower():
                                visi = f'[{v_d[VISI]}]'
                                str_nodes += '%-11s' % visi
                            # :\t{v_d["value"]}'
                            str_nodes += var.split(NS_SEP)[-1]
                    return str_nodes

                str_nodes = get_variables_info(str_nodes=str_nodes,
                                               node_data=node.data,
                                               io_type=TYPE_IN,
                                               disp_detail=display_variables)
                str_nodes = get_variables_info(str_nodes=str_nodes,
                                               node_data=node.data,
                                               io_type=TYPE_OUT,
                                               disp_detail=display_variables)
            for n in node.children:
                str_nodes += display_node(n, level,
                                          display_variables=display_variables)
            return str_nodes

        str_to_display = 'Nodes representation for Treeview ' + self.name + \
            display_node(self.root)
        return str_to_display

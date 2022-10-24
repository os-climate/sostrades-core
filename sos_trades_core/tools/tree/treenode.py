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
"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
"""
from json import dumps
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from os.path import dirname, isdir, isfile, join
import inspect
import os
import base64
import re
from os import listdir


class TreeNode:
    """
    Class to build tree node from data manager
    """
    STATUS_INPUT_DATA = 'INPUT_DATA'

    needed_variables = [SoSDiscipline.TYPE, SoSDiscipline.USER_LEVEL, SoSDiscipline.EDITABLE,
                        SoSDiscipline.COUPLING, SoSDiscipline.VALUE, SoSDiscipline.NUMERICAL, SoSDiscipline.OPTIONAL,
                        SoSDiscipline.CHECK_INTEGRITY_MSG]

    def __init__(self, name):
        """ class constructor
        """
        self.name = name

        # Children treenode: TreeNode[]
        self.children = []

        # Data coming from DataManager.data_dict for the associated namespace
        self.data = {}  # current node data

        # Store the discipline type
        self.node_type = 'data'

        # Treenode status (bijective with associated discipline if exist)
        self.status = TreeNode.STATUS_INPUT_DATA

        # Namespace associated to the disciplines
        self.full_namespace = ''

        # Disciplines indentifier (execution engine side => uuid)
        self.identifier = ''

        # Lisf of discipline identifier associated to this TreeNode
        self.disc_ids = []

        # Dict with addition of maturity for each discipline on this TreeNode
        self.multi_discipline_maturity = {}

        # Disciplines maturity (determined using the discipline maturity)
        self.maturity = ''

        # Data at discipline level (not namespace level => cf. self.data
        self.disc_data = {}  # to be able to show all variable at each discipline level


        # List of models present at current treenode
        self.models_full_path_list = []

        self.model_name = None
        self.model_name_full_path = None
        self.last_treenode = None

    def to_json(self):
        dict_obj = self.to_dict()
        return dumps(dict_obj)

    def to_dict(self):
        dict_obj = {}
        # Serialize name attribute
        dict_obj.update({'name': self.name})

        # Serialize data attribute
        dict_obj.update({'data': self.data})

        # Serialize node_type attribute
        dict_obj.update({'node_type': self.node_type})

        # Serialize status attribute
        dict_obj.update({'status': self.status})

        # Serialize full_namespace attribute
        dict_obj.update({'full_namespace': self.full_namespace})

        # Serialize identifier attribute
        dict_obj.update({'identifier': self.identifier})

        # Serialize maturity attribute
        dict_obj.update({'maturity': self.maturity})

        # Serialize disc_data attribute
        dict_obj.update({'disc_data': self.disc_data})

        # Serialize model_name_full_path attribute
        dict_obj.update({'model_name_full_path': self.model_name_full_path})

        # Serialize models_full_path_list attribute
        dict_obj.update({'models_full_path_list': self.models_full_path_list})

        # Serialize children attribute
        dict_child = []
        for tn in self.children:
            dict_child.append(tn.to_dict())
        dict_obj.update({'children': dict_child})
        return dict_obj

    def update_treenode_attributes(self, discipline, no_data=False, read_only=False):
        """ Inject discipline data into the current treenode

        :params: discipline to set into the treenode
        :type: SoSDiscipline
        """
        self.full_namespace = discipline.get_disc_full_name()
        self.identifier = discipline.disc_id
        self.disc_ids.append(self.identifier)
        self.node_type = discipline.__class__.__name__
        self.model_name_full_path = discipline.__module__
        self.models_full_path_list.append(discipline.__module__)

        if self.node_type != 'SoSCoupling':
            self.model_name = discipline.__module__.split('.')[-2]
        # Some modification has to be done on variable:
        # identifier : variable namespace + variable name
        # I/O type : 'in' for data_in and 'out' for data_out
        data_in = discipline.get_data_in()
        if not no_data:
            for key, data_key in data_in.items():
                namespaced_key = discipline.get_var_full_name(
                    key, data_in)
                new_disc_data = {
                    needed_key: data_key[needed_key] for needed_key in self.needed_variables}
                new_disc_data[SoSDiscipline.IO_TYPE] = SoSDiscipline.IO_TYPE_IN
                if read_only:
                    new_disc_data[SoSDiscipline.EDITABLE] = False
                new_disc_data["variable_key"] = self.create_data_key(self.model_name_full_path, SoSDiscipline.IO_TYPE_IN, key)
                self.update_disc_data(
                    new_disc_data, namespaced_key, discipline)

        data_out = discipline.get_data_out()
        if not no_data:
            for key, data_key in data_out.items():
                namespaced_key = discipline.get_var_full_name(
                    key, data_out)
                new_disc_data = {
                    needed_key: data_key[needed_key] for needed_key in self.needed_variables}
                new_disc_data[SoSDiscipline.IO_TYPE] = SoSDiscipline.IO_TYPE_OUT
                if read_only:
                    new_disc_data[SoSDiscipline.EDITABLE] = False
                new_disc_data["variable_key"] = self.create_data_key(self.model_name_full_path, SoSDiscipline.IO_TYPE_OUT, key)
                self.update_disc_data(
                    new_disc_data, namespaced_key, discipline)

        self.__manage_status(discipline.status)

        # Convert maturity dictionary to string for display purpose
        # Update multi_discipline_maturity with current discipline maturity
        o_mat = discipline.get_maturity()
        if isinstance(o_mat, dict):
            for k, v in o_mat.items():
                if k in self.multi_discipline_maturity:
                    self.multi_discipline_maturity[k] += v
                else:
                    self.multi_discipline_maturity[k] = v
        else:
            if len(o_mat) > 0:
                if o_mat in self.multi_discipline_maturity:
                    self.multi_discipline_maturity[o_mat] += 1
                else:
                    self.multi_discipline_maturity[o_mat] = 1

        if len(self.multi_discipline_maturity.items()) > 0:
            l_mat = [
                f'{k}({v})' for k, v in self.multi_discipline_maturity.items() if v != 0]
            s_mat = ' '.join(l_mat)
        else:
            s_mat = ''
        self.maturity = s_mat


    def create_data_key(self, disc_name, io_type, variable_name):
        io_type = io_type.lower()
        return f'{disc_name}_{io_type}put_{variable_name}'

    def update_disc_data(self, new_disc_data, namespace, discipline):
        """ Set variable from discipline into treenode disc_data
        :params: new_disc_data, variable data
        :type: SoSDiscipline variable data_dict

        :params: namespace, namespace of the variable
        :type: string

        :params: discipline to set into the treenode
        :type: SoSDiscipline
        """
        if namespace not in self.disc_data:
            self.disc_data[namespace] = new_disc_data
            self.disc_data[namespace][SoSDiscipline.DISCIPLINES_FULL_PATH_LIST] = [
                discipline.__module__]
        else:
            for key, value in new_disc_data.items():
                self.disc_data[namespace][key] = value
            if discipline.__module__ not in self.disc_data[namespace][SoSDiscipline.DISCIPLINES_FULL_PATH_LIST]:
                self.disc_data[namespace][SoSDiscipline.DISCIPLINES_FULL_PATH_LIST].append(
                    discipline.__module__)

    def __str__(self):
        children_str = ''.join([str(c) for c in self.children])
        return f'name : {self.name}, data : {self.data}, node_type : {self.node_type}, status : {self.status}, maturity: {self.maturity}\nchildren : {children_str}'

    def __manage_status(self, new_status):
        """ Each treenode can have multiple discipline
            for each new discipline, we compare the new status with the old one to choose
            which status is the most prioritary
        """
        current_status_priority = TreeNode.status_to_integer(self.status)
        new_status_priority = TreeNode.status_to_integer(new_status)

        if new_status_priority > current_status_priority:
            self.status = new_status

    @staticmethod
    def status_to_integer(status):
        """ Convert status to integer value to make some sort with status

            :params: status, disciplines status name
            :type: string

            :return: integer
        """

        if status == TreeNode.STATUS_INPUT_DATA:
            return -1
        elif status == SoSDiscipline.STATUS_VIRTUAL:
            return 0
        elif status == SoSDiscipline.STATUS_CONFIGURE:
            return 10
        elif status == SoSDiscipline.STATUS_DONE:
            return 20
        elif status == SoSDiscipline.STATUS_PENDING:
            return 30
        elif status == SoSDiscipline.STATUS_RUNNING:
            return 40
        else:  # status = SoSDiscipline.STATUS_FAILED
            return 50

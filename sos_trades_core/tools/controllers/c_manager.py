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
from copy import deepcopy


class CManager:
    """
    Padge Controller Manager
    Notation: 
        id = identifiers
        pt = pointer
    """
    ERROR_MSG = 'ERROR CManager.'

    #--constructor
    def __init__(self, FatherObj, gradient_active=True):
        self.__FatherObj = FatherObj
        self.__list_id = []
        self.__list_pt = []

        self.__gradient_active = True
        self.set_gradient_active(gradient_active)

    #--accessors
    def get_FatherObj(self):
        return self.__FatherObj

    def get_list_id(self):
        return self.__list_id

    def get_list_pt(self):
        return self.__list_pt

    #-- gradient active methods
    def get_gradient_active(self):
        return self.__gradient_active

    #-- Setters
    def set_gradient_active(self, gradient_active):
        if gradient_active != self.__gradient_active:
            self.__gradient_active = gradient_active
            if self.__gradient_active:
                for pt in self.get_list_pt():
                    pt.handle_dv_changes()
            self.flag_update_all()
# Problem no update function in this class
#            self.update()

    def flag_update_all(self):
        pt_list = self.get_list_pt()
        for pt in pt_list:
            pt.set_to_update()

    def flag_update_all_grad_arrays(self):
        pt_list = self.get_list_pt()
        for pt in pt_list:
            if pt.USE_GRADIENT_ARRAYS:
                pt.set_to_update()

    #--private methods
    def __repr__(self):
        info_string = self.get_print_header()
        if self.get_FatherObj() is not None:
            info_string += '\n   associated to FatherObj    : ' + self.get_FatherObj().get_tag()
        info_string += '\n   number of elements         : ' + \
            str(self.get_size())
        for pt in self.get_list_pt():
            info_string += pt.__repr__()
        info_string += self.get_print_footer()
        return info_string

    #--Methods
    def is_gradient_active(self):
        return self.__gradient_active

    def get_print_header(self):
        info_string = '\n-o0 CManager Information 0o-'
        return info_string

    def get_print_footer(self):
        info_string = '\n-o0 End of CManager Information 0o-'
        return info_string

    def get_size(self):
        return len(self.__list_id)

    def display(self):
        """
        display information about the manager
        """
        print(self)

    def add(self, pt, check=True):
        """
        add an element
        """
        ERROR_MSG = self.ERROR_MSG + 'add: '
        c_id = pt.get_id()
        if check:
            if c_id in self.__list_id:
                raise Exception(ERROR_MSG + 'Controller ' +
                                c_id + ' already exists!')

        self.__list_id.append(c_id)
        self.__list_pt.append(pt)

    def delete(self, Id):
        """
        delete an element
        """
        index = self.__list_id.index(Id)
        self.__list_pt[index].clean_dependencies()
        del self.__list_id[index]
        del self.__list_pt[index]

    def get_pt(self, Id, raise_error=True):
        """
        return pointer to element from base element id
        """
        ERROR_MSG = self.ERROR_MSG + 'get_pt: '
        try:
            index = self.__list_id.index(Id)
            return self.__list_pt[index]
        except:
            if raise_error:
                raise Exception(
                    ERROR_MSG + 'cannot find element ID=' + str(Id))
            else:
                return None

    def get_pt_from_index(self, index):
        """
        return pointer to element from manager index
        """
        try:
            return self.__list_pt[index]
        except:
            raise Exception('Unknown manager element of index : ' + str(index))

    def has_id(self, Id):
        return Id in self.get_list_id()

    def clean(self):
        id_list = deepcopy(self.get_list_id())
        for Id in id_list:
            self.delete(Id)

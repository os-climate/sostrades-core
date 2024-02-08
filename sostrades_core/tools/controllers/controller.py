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
Controller class: Can be variables, parameters, design variables, ...
"""

class Controller:
    """
    Controller class: Can be variables, parameters, design variables, ..
    """
    #--Class variables
    CLASS_MSG   = 'Controller'
    ERROR_MSG   = 'ERROR '+CLASS_MSG+'.'
    WARNING_MSG = 'WARNING '+CLASS_MSG+'.'
    USE_GRADIENT_ARRAYS = True #Declares if the class uses grad arrays that must be updated when design variables are created dynamically
    
    #--Constructor
    def __init__(self, Manager, Type, Id, check_add=True):
        ERROR_MSG=self.ERROR_MSG+'__init__: '
        self.__manager  = Manager    # a controller is linked to a Controller Manager 
        self.__type     = Type       # type of controller
        self.__id       = Id         # id of the controller
        self.__influences     = []
        self.__influences_id  = []
        self.__dependances    = []
        self.__dependances_id = []
        self.__is_updated=False # At least 1 update must be performed at instanciation
        self.__manager.add(self, check=check_add)#In order to have the right manager.get_ndv(), must be done before intializing self.__ndv
        
        self.__ndv          = self.__manager.get_ndv()

        if type(self.__id) not in [str, str]:
            raise Exception(ERROR_MSG+'Controller ID must a string')
        
        self.specific_init()
        self.handle_dv_changes()
        
    def specific_init(self):
        pass
        
    def __repr__(self):
        info_string =  '\n----------------------------------------------'
        info_string += '\n   ID              : '+self.get_id()
        info_string += '\n   Type            : '+self.get_type()
        return info_string

    #--Accessors
    def get_manager(self):
        return self.__manager
    
    def is_gradient_active(self):
        return self.get_manager().is_gradient_active()
    
    def get_type(self):
        return self.__type
    
    def get_id(self):
        return self.__id
    
    def get_ndv(self):
        return self.__ndv

    #Dependancies / Influencies management
    def get_dependances(self):
        return self.__dependances
    
    def get_dependances_id(self):
        return self.__dependances_id
    
    def get_influences(self):
        return self.__influences
    
    def is_influent(self):
        return len(self.__influences)>0
    
    def get_influences_id(self):
        return self.__influences_id 
    
    def _add_influence(self,controller):
        """
        influences are managed via dependances, user should not add any dependance to the list.
        """
        if controller is not None:
            cid = controller.get_id()
            #if cid not in self.get_influences_id():
            self.get_influences().append(controller)
            self.get_influences_id().append(cid)
        
    def add_dependance(self,controller):
        if controller is not None:
            cid = controller.get_id()
            #if cid not in self.get_dependances_id():
            self.get_dependances().append(controller)
            self.get_dependances_id().append(cid)
            controller._add_influence(self)
        
    def special_dependance_update(self,pt):
        return 
        #WARNING_MSG=self.WARNING_MSG+'special_dependance_update: '
        #print WARNING_MSG+'element ID='+self.get_id()+': unexpected call!!! please contact a padge developper!!!'
        
    def del_dependance(self,controller):
        if controller is not None:
            cid = controller.get_id()
            #if cid in self.get_dependances_id():
            self.get_dependances().remove(controller)
            self.get_dependances_id().remove(cid)
            controller.del_influence(self)
        
    def del_influence(self,controller):
        if controller is not None:
            cid = controller.get_id()
            #if cid in self.get_influences_id():
            self.get_influences().remove(controller)
            self.get_influences_id().remove(cid)
            
    def is_updated(self):
        return self.__is_updated
    
    def is_to_update(self):
        return not self.is_updated() and self.dependances_updated()
    
    def set_to_update(self):
        self.set_controllers_to_update()
        self.__is_updated=False
        
    def set_controllers_to_update(self):
        for dep_ctrl in self.get_influences():
            if dep_ctrl.is_updated():
                dep_ctrl.set_to_update()
                dep_ctrl.set_controllers_to_update()
        
    def set_updated(self):
        self.__is_updated=True
        
    def dependances_updated(self):
        for pt in self.get_dependances():
            if not pt.is_updated():
                return False
        return True
        
    #--Methods
    def clean_dependencies(self):
        for pt in self.get_dependances():
            pt.del_influence(self)
        for pt in self.get_influences():
            pt.del_dependance(self)
    
    def display(self):
        print(self)
        
    def reset_ndv(self):
        self.__ndv = self.get_manager().get_ndv()
        
    def is_ndv_up_to_date(self):
        return self.get_ndv() == self.get_manager().get_ndv()
    
    def handle_dv_changes(self):
        """
        Handles any change in DV number. At least, reallocate gradients tables
        """
    
    def check(self):
        pass
    
    def update_specific(self):
        """
        Update specific to each Controller.
        """
    
    def update(self):
        self.check()
        if not self.is_ndv_up_to_date():
            self.reset_ndv()
            if self.is_gradient_active():
                self.handle_dv_changes()
        
        self.update_specific()
        self.set_updated()




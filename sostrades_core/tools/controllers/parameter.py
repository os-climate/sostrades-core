'''
Copyright 2022 Airbus SAS
Modifications on 2024/05/16 Copyright 2024 Capgemini

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
from re import compile

from .base_controller import BaseController

# from MDOTools.Controllers.Formula.Formula import Formula
from .simpy_formula import SympyFormula as Formula


class Parameter(BaseController):
    """
    PParameter Class
    """
    #--Class Variables
    CLASS_MSG = 'PParameter'
    ERROR_MSG = 'ERROR '+CLASS_MSG+'.'
    
    #--Constructor
    def __init__(self,PBCManager,Id,fexpr,AliasDict=None, namespace=None):

        BaseController.__init__(self, PBCManager, Id, value=0., BCType='Parameter')
        
        self.__fexpr       = None
        self.__AliasDict   = None
        self.__FormulaObj  = None
        self.__dep_id_list = []
        self.__dep_dict    = {}
        self.new_id_list   = None
        self.namespace = namespace
        
        self.set_fexpr(fexpr,AliasDict=AliasDict)
        
    def __repr__(self):
        """
        Display some information about the variable
        """
        info_string =BaseController.__repr__(self)
        info_string+="\n   Formula         : "+self.__FormulaObj.get_formula()
        info_string+="\n   Grad Formula    : "+self.__FormulaObj.get_grad_formula()
        info_string+="\n   Value           :%24.16e"%self.get_value()
        info_string+="\n   Gradient        : "+str(self.get_gradient())
        return info_string
    
    def __replace_aliases(self):
        BC_manager=self.get_manager()
        if self.__AliasDict is not None:
            alias_keys = list(self.__AliasDict.keys())
            sep = compile('([,\^\*\+\[\]/\(\)-])')
            fexpr = sep.split(self.__fexpr)
            for indice in range(len(fexpr)):
                if fexpr[indice].strip() in alias_keys:
                    fexpr[indice] = self.__AliasDict[fexpr[indice].strip()]
            self.__fexpr=str.join('',fexpr)
        if self.namespace is not None:
            self.__FormulaObj  = Formula(self.__fexpr, fgrad=False) 
            dep_id_list = self.__FormulaObj.get_token_list()
            prefix = self.namespace+'.'
            sep = compile('([,\^\*\+\[\]/\(\)-])')
            fexpr = sep.split(self.__fexpr)
            for indice in range(len(fexpr)):
                if fexpr[indice] in dep_id_list:
                    pt=BC_manager.get_pt(fexpr[indice], raise_error=False)
                    if pt is None:
                        fexpr[indice] = prefix+fexpr[indice]
            self.__fexpr=str.join('',fexpr)
            
    def set_and_check_deps(self):
        self.__update_dep_list()
        
    def set_fexpr(self,fexpr,AliasDict=None):
        self.__fexpr       = fexpr
        self.__AliasDict   = AliasDict
        
        self.__replace_aliases()
        
        fgrad=self.is_gradient_active()
        
        self.__FormulaObj  = Formula(self.__fexpr, fgrad=fgrad) 
        
        self.new_id_list = self.__FormulaObj.get_token_list()
        
    def get_expr(self):
        return self.__fexpr
        
    def get_formula(self):
        return self.__FormulaObj.get_formula()
    
    def get_grad_formula(self):
        return self.__FormulaObj.get_grad_formula()
    
    def __update_dep_list(self):
        BC_manager=self.get_manager()
        
        for Id in self.__dep_id_list:
            if Id not in self.new_id_list:
                pt=BC_manager.get_pt(Id)
                self.del_dependance(pt)
                del self.__dep_dict[Id]
                
        for Id in self.new_id_list:
            if Id not in self.__dep_id_list:
                try:
                    pt=BC_manager.get_pt(Id)
                except:
                    print(BC_manager)
                    print(self, list(self.new_id_list))
                    raise Exception('Error in Parameter: '+str(self.get_id())+' cannot find '+str(Id)+'!')
                self.add_dependance(pt)
                self.__dep_dict[Id]=pt
            
        self.__dep_id_list=self.new_id_list
        
    def special_dependance_update(self,pt):
        ERROR_MSG=self.ERROR_MSG+'special_dependance_update: '
        Id=pt.get_id()
        if Id not in self.__dep_id_list:
            raise Exception(ERROR_MSG+'method cannot be applied to non existing id ('+str(Id)+')')
        self.add_dependance(pt)
        self.set_to_update()
        self.__dep_dict[Id]=pt
    
    def update_specific(self):
        eval_dict={}
        for Id in self.__dep_id_list:
            eval_dict[Id]     = self.__dep_dict[Id].get_value()
        if self.is_gradient_active():
            self.__FormulaObj.set_grad(fgrad=True)
            for Id in self.__dep_id_list:
                eval_dict['d'+Id] = self.__dep_dict[Id].get_gradient()
        else:
            self.__FormulaObj.set_grad(fgrad=False)
        
        try:
            self.__FormulaObj.evaluate(eval_dict)
        except:
            ERROR_MSG=self.ERROR_MSG+' Failed to evaluate parameter '+self.get_id()+' of expression: '+self.__fexpr
            raise Exception(ERROR_MSG)
        
        value    = self.__FormulaObj.get_value()
        self.set_value(value,flag_updates=False)
        
        if self.is_gradient_active():
            gradient = self.__FormulaObj.get_gradient()
            self.set_gradient(gradient)

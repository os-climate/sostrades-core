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
from sos_trades_core.study_manager.study_manager import StudyManager
import pandas as pd
from sos_trades_core.execution_engine.func_manager.func_manager import FunctionManager


class Study(StudyManager):

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):

        EQ_CONSTRAINT = FunctionManager.EQ_CONSTRAINT
        OBJECTIVE = FunctionManager.OBJECTIVE
        ns = f'{self.study_name}'
        sc_name = "SellarOptimScenario"
        dspace_dict = {'variable': ['y_2', 'y_1'],
                       'value': [6., 6.],
                       'lower_bnd': [0., 0.],
                       'upper_bnd': [10., 10.],
                       'enable_variable': [True, True],
                       'activated_elem': [[True], [True, True]]}
#                   'type' : ['float',['float','float'],'float','float']
        dspace = pd.DataFrame(dspace_dict)

        disc_dict = {}
        disc_dict[f'{ns}.SellarOptimScenario.SellarCoupling.sub_mda_class'] = 'MDAGaussSeidel'
        disc_dict[f'{ns}.SellarOptimScenario.SellarCoupling.max_mda_iter'] = 4

        disc_dict[f'{ns}.SellarOptimScenario.SellarCoupling.y_1'] = [6.]
        # Optim inputs
        disc_dict[f'{ns}.SellarOptimScenario.max_iter'] = 500
        disc_dict[f'{ns}.SellarOptimScenario.algo'] = "SLSQP"
        disc_dict[f'usecase.SellarOptimScenario.design_space'] = dspace
        # TODO: what's wrong with IDF
        disc_dict[f'{ns}.SellarOptimScenario.formulation'] = 'DisciplinaryOpt'
        # f'{ns}.SellarOptimScenario.obj'
        disc_dict[f'{ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{ns}.SellarOptimScenario.ineq_constraints'] = [
        ]

        disc_dict[f'{ns}.SellarOptimScenario.eq_constraints'] = ['c_3', 'c_4'
                                                                 ]
        # f'{ns}.SellarOptimScenario.c_1', f'{ns}.SellarOptimScenario.c_2']

        disc_dict[f'{ns}.SellarOptimScenario.algo_options'] = {
            #"maxls": 6,
            #"maxcor": 3,
            #"ftol_rel": 1e-10,
        }
        """
        # Sellar inputs
        disc_dict[f'{ns}.{sc_name}.y_01'] = array([100.])
        disc_dict[f'{ns}.{sc_name}.y_02'] = array([100.])

        func_df = pd.DataFrame(columns=['variable', 'ftype', 'weight'])
        func_df['variable'] = ['c_3', 'c_4', 'obj']
        func_df['ftype'] = [EQ_CONSTRAINT, EQ_CONSTRAINT, OBJECTIVE]
        func_df['weight'] = [10000, 10000, 0.011]
        func_mng_name = 'FunctionManager'

        prefix = self.study_name + '.SellarOptimScenario.' + func_mng_name + '.'
        values_dict = {}
        values_dict[prefix +
                    FunctionManagerDisc.FUNC_DF] = func_df

        disc_dict.update(values_dict)
        """
        return [disc_dict]


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes(display_variables=True)
    # uc_cls.execution_engine.root_process.sos_disciplines[0].sos_disciplines[0].coupling_structure.graph.export_initial_graph(
    #    'init.pdf')
    uc_cls.run()
    """
    uc_cls.execution_engine.root_process.coupling_structure.plot_n2_chart(
        'n2.png')
    """

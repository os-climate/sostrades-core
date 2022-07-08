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
from numpy import array
import pandas as pd
from sos_trades_core.tools.post_processing.post_processing_factory import PostProcessingFactory


class Study(StudyManager):

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):
        ns = f'{self.study_name}'
        sc_name = "SellarOptimScenario"
        c_name = "SellarCoupling"
        dspace_dict = {'variable': ['x', 'z'],
                       'value': [[1.], [5., 2.]],
                       'lower_bnd': [[0.], [-10., 0.]],
                       'upper_bnd': [[10.], [10., 10.]],
                       'enable_variable': [True, True],
                       'activated_elem': [[True], [True, True]]}
#                   'type' : ['float',['float','float'],'float','float']
        dspace = pd.DataFrame(dspace_dict)

        disc_dict = {}
        # Optim inputs
        disc_dict[f'{ns}.SellarOptimScenario.max_iter'] = 100
        # SLSQP, NLOPT_SLSQP
        disc_dict[f'{ns}.SellarOptimScenario.algo'] = "SLSQP"
        disc_dict[f'{ns}.SellarOptimScenario.design_space'] = dspace
        # TODO: what's wrong with IDF
        disc_dict[f'{ns}.SellarOptimScenario.formulation'] = 'DisciplinaryOpt'
        # f'{ns}.SellarOptimScenario.obj'
        disc_dict[f'{ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{ns}.SellarOptimScenario.ineq_constraints'] = [
            'c_1', 'c_2']
        # f'{ns}.SellarOptimScenario.c_1', f'{ns}.SellarOptimScenario.c_2']

        disc_dict[f'{ns}.SellarOptimScenario.algo_options'] = {"ftol_rel": 1e-10,
                                                               "ineq_tolerance": 2e-3,
                                                               "normalize_design_space": False}

        # Sellar inputs
        disc_dict[f'{ns}.{sc_name}.{c_name}.x'] = array([1.])
        disc_dict[f'{ns}.{sc_name}.{c_name}.y_1'] = array([1.])
        disc_dict[f'{ns}.{sc_name}.{c_name}.y_2'] = array([1.])
        disc_dict[f'{ns}.{sc_name}.{c_name}.z'] = array([1., 1.])
        disc_dict[f'{ns}.{sc_name}.{c_name}.Sellar_Problem.local_dv'] = 10.

        return [disc_dict]


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes(display_variables=True)

    uc_cls.run()

#    ppf = PostProcessingFactory()
#    disc = uc_cls.execution_engine.dm.get_disciplines_with_name(
#        f'{uc_cls.study_name}.SellarOptimScenario.SellarCoupling')
#    filters = ppf.get_post_processing_filters_by_discipline(
#        disc[0])
#    graph_list = ppf.get_post_processing_by_discipline(
#        disc[0], filters, as_json=False)
#
#    for graph in graph_list:
#        graph.to_plotly().show()

    # Uncomment to see dependancy graphs
#     uc_cls.execution_engine.root_process.coupling_structure.graph.export_initial_graph(
#             "whole_process_initial.pdf")
    # XDSMize test
#     uc_cls.execution_engine.root_process.sos_disciplines[0].xdsmize()
    # to visualize in an internet browser :
    # - download XDSMjs at https://github.com/OneraHub/XDSMjs and unzip
    # - replace existing xdsm.json inside by yours
    # - in the same folder, type in terminal 'python -m http.server 8080'
    # - open in browser http://localhost:8080/xdsm.html

#     uc_cls.execution_engine.root_process.sos_disciplines[0].sos_disciplines[0].coupling_structure.graph.export_initial_graph(
#             "initial.pdf")
#     uc_cls.execution_engine.root_process.sos_disciplines[0].sos_disciplines[0].coupling_structure.graph.export_reduced_graph(
#             "reduced.pdf")

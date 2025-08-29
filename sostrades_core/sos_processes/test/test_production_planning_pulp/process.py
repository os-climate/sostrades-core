#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Planning Optimization Process Builder

Process for multi-product manufacturing optimization using PuLP linear programming
"""

from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):
    """Process builder for production planning optimization using PuLP"""

    # ontology information
    _ontology_data = {
        'label': 'Production Planning Optimization with PuLP',
        'description': 'Multi-product manufacturing optimization using PuLP linear programming',
        'category': 'Operations Research',
        'version': '1.0',
    }

    def get_builders(self):
        '''
        Create optimization process with production planning discipline
        '''
        # Define discipline module path
        disc_dir = 'sostrades_core.sos_processes.test.test_production_planning_pulp.'

        # Modules dictionary - production planning discipline
        mods_dict = {
            'ProductionPlanningProblem': disc_dir + 'production_planning_problem.ProductionPlanningProblem',
        }

        # Namespace dictionary
        ns_dict = {
            'ns_production': self.ee.study_name + '.ProductionPlanningOptimScenario.ProductionCoupling',
        }

        # Create builder list for the discipline
        builder_list = self.create_builder_list(mods_dict, ns_dict=ns_dict)

        # Create coupling builder
        coupling_builder = self.ee.factory.create_builder_coupling("ProductionCoupling")
        coupling_builder.set_builder_info('cls_builder', builder_list)

        # Create optimization builder
        opt_builder = self.ee.factory.create_optim_builder(
            'ProductionPlanningOptimScenario', [coupling_builder])

        return opt_builder

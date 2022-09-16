# test test_disc_hessian_vs_MS_from_proc 
This "test_disc_hessian_vs_MS_from_proc" specifies an example of a very symple multi scenario process.
It uses a scatter based on :


1) a scenario build map

		scenario_map = 
				{'input_name': 'scenario_list',
				'input_ns': 'ns_scatter_scenario',
				'output_name': 'scenario_name',
				'scatter_ns': 'ns_scenario',
				'gather_ns': 'ns_scatter_scenario',
				'ns_to_update': []}
				
- the namespace table 
				'ns_scatter_scenario' = f'{self.ee.study_name}.vs_MS'
				'ns_scenario' : not needed
				
2) the process 'test_disc_hessian' based on 
- the discipline test_discs.disc_hessian.DiscHessian

    DESC_IN = {
        'x': {'type': 'float', 'unit': '-'},
        'y': {'type': 'float', 'unit': '-'},
        'ax2': {'type': 'float', 'unit': '-'},
        'by2': {'type': 'float', 'unit': '-'},
        'cx': {'type': 'float', 'unit': '-'},
        'dy': {'type': 'float', 'unit': '-'},
        'exy': {'type': 'float', 'unit': '-'},
    }
    DESC_OUT = {
        'z': {'type': 'float', 'unit': '-'}
	

		'z' =  'ax2' * ('x'**2) + 'by2' * ('y'**2) + 'cx' * 'x' + 'dy' * 'y' + 'exy' * ('x' * 'y')

# Test (disc 1 scenario, disc 3 scenario) coupling
A coupling based on:

1) the discipline discipline test_discs.disc1_scenario.Disc1

		DESC_IN = 
				{'x': {'type': 'float', 'unit': '-', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_data_ac'},
				'a': {'type': 'float', 'unit': '-', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_data_ac'},
				'b': {'type': 'float', 'unit': '-'}}

		DESC_OUT =
				{'indicator': {'type': 'float', 'unit': '-'},
				'y': {'type': 'float', 'unit': '-', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ac'}}
	
		'indicator' = a * b
		'y': a * x + b
- the namespace table 

				'ns_ac' = self.ee.study_name
				'ns_data_ac' =  self.ee.study_name


2) the discipline disc3_scenario.Disc3

		DESC_IN = 
				{'z': {'type': 'float', 'unit': '-', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_disc3'},
				'constant': {'type': 'float', 'unit': '-'},
				'power': {'type': 'int', 'unit': '-'}}

		DESC_OUT =
				{'o': {'type': 'float', 'unit': '-', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_out_disc3'}}
	
		'o'=  constant + z**power
		
- the namespace table 

				'ns_disc3' = f'{self.ee.study_name}.Disc3
				'ns_out_disc3'=  f'{self.ee.study_name}'
# Discipline disc1_scenario.Disc1
This disc1_scenario.Disc1 discipline is specified as follows:

	DESC_IN = 
			{'x': {'type': 'float', 'unit': '-', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_data_ac'},
			'a': {'type': 'float', 'unit': '-', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_data_ac'},
			'b': {'type': 'float', 'unit': '-'}}

	DESC_OUT =
			{'indicator': {'type': 'float', 'unit': '-'},
			'y': {'type': 'float', 'unit': '-', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ac'}}

	'indicator' = a * b
	'y': a * x + b

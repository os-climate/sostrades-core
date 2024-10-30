'''
Copyright 2024 Capgemini

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
import numpy as np
import pandas as pd
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from energy_models.glossaryenergy import GlossaryEnergy
from climateeconomics.glossarycore import GlossaryCore

"""
This script is used to test the loading feature
"""

year_start = 2020
year_end = 2100
years = np.arange(year_start, year_end + 1)
invest_year_start = 20.
initial_production = 6.

name = 'Test'
model_name = GlossaryEnergy.Hydropower
ee = ExecutionEngine(name)
ns_dict = {'ns_public': name,
           'ns_energy': name,
           'ns_energy_study': f'{name}',
           'ns_electricity': name,
           'ns_resource': name}
ee.ns_manager.add_ns_def(ns_dict)

mod_path = 'energy_models.models.electricity.hydropower.hydropower_disc.HydropowerDiscipline'
builder = ee.factory.get_builder_from_module(
    model_name, mod_path)

ee.factory.set_builders_to_coupling_builder(builder)

ee.configure()
ee.display_treeview_nodes()



def run_model(x: list, year_end: int = year_end):
    init_age_distrib_factor = x[0]
    invest_years_optim = x[1:]
    invest_df = pd.DataFrame({GlossaryEnergy.Years: years,
                                                          GlossaryCore.InvestValue: list(invest_years_optim)})

    inputs_dict = {
        f'{name}.{GlossaryEnergy.YearStart}': year_start,
        f'{name}.{GlossaryEnergy.YearEnd}': year_end,
        f'{name}.{model_name}.{GlossaryEnergy.InvestLevelValue}': invest_df,
        f'{name}.{GlossaryEnergy.CO2TaxesValue}': pd.DataFrame(
            {GlossaryEnergy.Years: years, GlossaryEnergy.CO2Tax: np.linspace(0., 0., len(years))}),
        f'{name}.{GlossaryEnergy.StreamsCO2EmissionsValue}': pd.DataFrame({GlossaryEnergy.Years: years}),
        f'{name}.{GlossaryEnergy.StreamPricesValue}': pd.DataFrame({GlossaryEnergy.Years: years}),
        f'{name}.{GlossaryEnergy.ResourcesPriceValue}': pd.DataFrame({GlossaryEnergy.Years: years}),
        f'{name}.{GlossaryEnergy.TransportCostValue}': pd.DataFrame({GlossaryEnergy.Years: years, 'transport': np.zeros(len(years))}),
        f'{name}.{model_name}.{GlossaryEnergy.InitialPlantsAgeDistribFactor}': init_age_distrib_factor,
        f'{name}.{model_name}.initial_production': initial_production
    }
    # bug: must load the study twice so that modifications are taked into accout
    ee.load_study_from_input_dict(inputs_dict)

    ee.execute()

    prod_df = ee.dm.get_value(ee.dm.get_all_namespaces_from_var_name(GlossaryEnergy.TechnoProductionValue)[0]) #PWh
    init_prod = ee.dm.get_value(ee.dm.get_all_namespaces_from_var_name('initial_production')[0])

    return prod_df, init_prod



# Initial guess for the variables invest from year 2025 to 2100.
x0 = np.concatenate((np.array([1.0]), invest_year_start * np.ones(len(years))))
output = []
for iteration in range(0, 4):
    prod_df, init_prod = run_model(list(x0))
    output.append(f"#iteration {iteration}: initial_production requested={initial_production} | initial_production actually used={init_prod} | inital_production_computed={prod_df['electricity (TWh)'][0]}")

for message in output:
    print(message)
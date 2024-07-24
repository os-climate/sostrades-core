
> [!IMPORTANT]
> On June 26 2024, Linux Foundation announced the merger of its financial services umbrella, the Fintech Open Source Foundation ([FINOS](https://finos.org)), with OS-Climate, an open source community dedicated to building data technologies, modeling, and analytic tools that will drive global capital flows into climate change mitigation and resilience; OS-Climate projects are in the process of transitioning to the [FINOS governance framework](https://community.finos.org/docs/governance); read more on [finos.org/press/finos-join-forces-os-open-source-climate-sustainability-esg](https://finos.org/press/finos-join-forces-os-open-source-climate-sustainability-esg)


# SoSTrades_CORE


## Description
SoSTradesCore is the Python package containing the core of the execution engine of System of Systems Trades

## Packages installation
pip install -r requirements.in --trusted-host pypi.org --trusted-host files.pythonhosted.org

A dedicated version of GEMSEO is required, for now:
https://gitlab.com/sostrades/gemseo branch sos_develop


## Overview
This package is divided into 6 parts:

- execution_engine: contains all the methods and wrapped class of gemseo tools necessary to implement processes and studies in SoSTrades
- sos_processes: contains test processes built with disciplines from sos_wrapping
- sos_wrapping: contains test disciplines covering execution engine functionalities
- study_manager: contains generic class to implement studies
- tests: contains tests on execution_engine functionalities, based on sos_processes and sos_wrapping
- tools: contains generic tools of execution_engine

## License
The sostrades-core source code is distributed under the Apache License Version 2.0.
A copy of it can be found in the LICENSE file.

The sostrades-core product depends on other software which have various licenses.
The list of dependencies with their licenses is given in the CREDITS.rst file.

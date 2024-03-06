# SoSTrades platform overview

The SoSTrades platform is a cloud based collaborative generic simulation platform​.\
You can access it by following this [link](https://validation.osc-tsa.com/) :

[![homepage](../images/platform_logo.png)](https://validation.osc-tsa.com/ "Access open SoSTrades simulation platform")

> *Note: The connection to the platform is free but you will be asked to use a GitHub account.*

## A generic platform ​adaptable to the simulation needs ​of specific projects​

* Designed to handle long-term trajectories 2030, 2050, 2100 unlike industrial digital twins or classic “System of Systems” approaches​ 
* Provides all the state of the art in mathematics for simulation: ​
>* Design of experiments
>* Sensitivity analyses
>* Optimization
>* Uncertainties
* Cloud native and scalable, ​able to handle 300,000+ coupled variables (demonstrated on the WITNESS model)​
* Automatically interconnect models and orchestrate studies​

## An open source platform ​that enables multi-user collaboration​

* Collaborative by design, so that multiple actors can cooperate live defining assumptions and analyzing results​
* Open-source within the framework of the “Open Source for Climate” project of the Linux Foundation: sovereignty, adaptability, mutualized non-differentiating evolutions​
* Web interface, all you need is an internet link to use it, no installation​
* Data confidentiality is ensured by individual access rights and scenarios encryption on disks

# SoSTrades-core overview

The SoSTrades platform is based on a core execution Python module that is named **sostrades-core**.\
Although **sostrades-cor**e is meant to work with the SoSTrades platform, you can use it in scripted mode.\
This module allows to configure and execute generic processes and theirs associated tests directly using Python in a terminal.\
**sostrades-core** is thus the entry point for developpers that want to build a new simulation process and use it through the GUI afterwards.

> *Note: SoSTrades leverages the GEMSEO library's numerical features, including automatic coupling and many iterative solvers.* [For more about GEMESO](https://gemseo.readthedocs.io/en/stable/)


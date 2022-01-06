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
from gemseo.core.dependency_graph import DependencyGraph
 
def get_execution_sequence(cls):
    """Compute the execution sequence of the disciplines.
 
    Returns:
        list(set(tuple(set(MDODisciplines))))
    """
    condensed_graph_func = getattr(cls, "_DependencyGraph__create_condensed_graph")
    condensed_graph = condensed_graph_func()
    execution_sequence = []
 
    while True:
        get_leaves_func = getattr(cls, "_DependencyGraph__get_leaves")
        leaves = get_leaves_func(condensed_graph)
 
        if not leaves:
            break
 
        # SoSTrades fix : use list() instead of set() to preserve disciplines ordering
        parallel_tasks = list(
            tuple(condensed_graph.nodes[node_id]["members"]) for node_id in leaves
        )
        execution_sequence += [parallel_tasks]
        condensed_graph.remove_nodes_from(leaves)
 
    return list(reversed(execution_sequence))
 
setattr(DependencyGraph, "get_execution_sequence", get_execution_sequence)

from infinigen.core.constraints.example_solver import (
    Solver,
    greedy,
    populate,
    state_def,
)
from .basic_scene import all_vars


def init_physcene(stages,limits,solver,state,p):
   # region init large physcene
    def init_graph_physcene(this_stage):
        assignments = greedy.iterate_assignments(
            stages["on_floor"], state, all_vars, limits, nonempty=True
        )
        for i, vars in enumerate(assignments):
            solver.init_graph_physcene(
            # solver.init_graph_gpt(
                # stages["on_floor"],
                var_assignments=vars,
                stage=this_stage,
            )
        return solver.state

    state = p.run_stage(
        "init_graph_physcene", init_graph_physcene, this_stage="large", use_chance=False, default=state
    )
    # endregion
    return state,solver

def init_gpt(stages,limits,solver,state,p):
   # region init large physcene
    def init_graph_gpt(this_stage):
        assignments = greedy.iterate_assignments(
            stages["on_floor"], state, all_vars, limits, nonempty=True
        )
        for i, vars in enumerate(assignments):
            solver.init_graph_gpt(
            # solver.init_graph_gpt(
                # stages["on_floor"],
                var_assignments=vars,
                stage=this_stage,
            )
        return solver.state

    state = p.run_stage(
        "init_graph_gpt", init_graph_gpt, this_stage="large", use_chance=False, default=state
    )
    state = p.run_stage(
        "init_graph_gpt", init_graph_gpt, this_stage="medium", use_chance=False, default=state
    )

    # state = p.run_stage(
    #     "init_graph", init_graph, this_stage="small", use_chance=False, default=state
    # )
    # endregion
    return state,solver


def init_metascene(stages,limits,solver,state,p):
   # region init large physcene
    def init_graph_metascene(this_stage):
        assignments = greedy.iterate_assignments(
            stages["on_floor"], state, all_vars, limits, nonempty=True
        )
        for i, vars in enumerate(assignments):
            solver.init_graph_metascene(
            # solver.init_graph_gpt(
                # stages["on_floor"],
                var_assignments=vars,
                stage=this_stage,
            )
        return solver.state

    state = p.run_stage(
        "init_graph_metascene", init_graph_metascene, this_stage="large", use_chance=False, default=state
    )
    # endregion
    return state,solver
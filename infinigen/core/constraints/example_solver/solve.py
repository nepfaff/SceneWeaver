# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import copy
import logging
from pathlib import Path

import bpy
import gin
import numpy as np
from tqdm import trange

# from debug import invisible_others, visible_others
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints import reasoning as r
from infinigen.core.constraints.evaluator import domain_contains
from infinigen.core.constraints.example_solver import (
    greedy,
    propose_continous,
    propose_discrete,
)
from infinigen.core.constraints.example_solver.propose_discrete import moves
from infinigen.core.constraints.example_solver.state_def import State
from infinigen.core.util import blender as butil

from .annealing import SimulatedAnnealingSolver
from .room import MultistoryRoomSolver, RoomSolver

from infinigen.assets.objects import (
            appliances,
            bathroom,
            decor,
            elements,
            lamp,
            seating,
            shelves,
            table_decorations,
            tables,
            tableware,
            wall_decorations,
        )
from infinigen.core.constraints import usage_lookup
import math
from . import moves, propose_relations, state_def

import importlib
from infinigen.core import tags as t

logger = logging.getLogger(__name__)

GLOBAL_GENERATOR_SINGLETON_CACHE = {}


def map_range(x, xmin, xmax, ymin, ymax, exp=1):
    if x < xmin:
        return ymin
    if x > xmax:
        return ymax

    t = (x - xmin) / (xmax - xmin)
    return ymin + (ymax - ymin) * t**exp


@gin.register
class LinearDecaySchedule:
    def __init__(self, start, end, pct_duration):
        self.start = start
        self.end = end
        self.pct_duration = pct_duration

    def __call__(self, t):
        return map_range(t, 0, self.pct_duration, self.start, self.end)


@gin.configurable
class Solver:
    def __init__(
        self,
        output_folder: Path,
        multistory: bool = False,
        restrict_moves: list = None,
        addition_weight_scalar: float = 1.0,
    ):
        """Initialize the solver

        Parameters
        ----------
        output_folder : Path
            The folder to save output plots to
        print_report_freq : int
            How often to print loss reports
        multistory : bool
            Whether to use the multistory room solver
        constraints_greedy_unsatisfied : str | None
            What do we do if relevant constraints are unsatisfied at the end of a greedy stage?
            Options are 'warn` or `abort` or None

        """

        self.output_folder = output_folder

        self.optim = SimulatedAnnealingSolver(
            output_folder=output_folder,
        )

        self.room_solver_fn = MultistoryRoomSolver if multistory else RoomSolver
        self.state: State = None
        self.all_roomtypes = None
        self.dimensions = None

        self.moves = self._configure_move_weights(
            restrict_moves, addition_weight_scalar=addition_weight_scalar
        )

    def _configure_move_weights(self, restrict_moves, addition_weight_scalar=1.0):
        schedules = {
            "addition": (
                propose_discrete.propose_addition,
                LinearDecaySchedule(
                    6 * addition_weight_scalar, 0.1 * addition_weight_scalar, 0.9
                ),
            ),
            "deletion": (
                propose_discrete.propose_deletion,
                LinearDecaySchedule(2, 0.0, 0.5),
            ),
            "plane_change": (
                propose_discrete.propose_relation_plane_change,
                LinearDecaySchedule(2, 0.1, 1),
            ),
            "resample_asset": (
                propose_discrete.propose_resample,
                LinearDecaySchedule(1, 0.1, 0.7),
            ),
            "reinit_pose": (
                propose_continous.propose_reinit_pose,
                LinearDecaySchedule(1, 0.5, 1),
            ),
            "translate": (propose_continous.propose_translate, 1),
            "rotate": (propose_continous.propose_rotate, 0.5),
        }

        if restrict_moves is not None:
            schedules = {k: v for k, v in schedules.items() if k in restrict_moves}
            logger.info(
                f"Restricting {self.__class__.__name__} moves to {list(schedules.keys())}"
            )

        return schedules

    @gin.configurable
    def choose_move_type(
        self,
        it: int,
        max_it: int,
    ):
        t = it / max_it
        names, confs = zip(*self.moves.items())
        funcs, scheds = zip(*confs)
        weights = np.array([s if isinstance(s, (float, int)) else s(t) for s in scheds])
        return np.random.choice(funcs, p=weights / weights.sum())

    def solve_rooms(self, scene_seed, consgraph: cl.Problem, filter: r.Domain):
        self.state, self.all_roomtypes, self.dimensions = self.room_solver_fn(
            scene_seed
        ).solve()
        return self.state

    @gin.configurable
    def solve_objects(
        self,
        consgraph: cl.Problem,
        filter_domain: r.Domain,
        var_assignments: dict[str, str],
        n_steps: int,
        desc: str,
        abort_unsatisfied: bool = False,
        print_bounds: bool = False,
        expand_collision: bool = False,
    ):
        filter_domain = copy.deepcopy(filter_domain)
        """
        Domain({Semantics.Object, -Semantics.Room}, [
            (StableAgainst({}, {Subpart.SupportSurface, Subpart.Visible, -Subpart.Ceiling, -Subpart.Wall}), Domain({Semantics.Bathroom, Variable(room), Semantics.Room, -Semantics.Object}, [])),
            (-AnyRelation(), Domain({Semantics.Object, -Semantics.Room}, []))
        ])
        """
        # if desc == "side_obj_0":
        #     import pdb
        #     pdb.set_trace()
        desc_full = (desc, *var_assignments.values())
        # ('on_floor_0', 'bathroom_0-0')

        dom_assignments = {
            k: r.Domain(self.state.objs[objkey].tags)
            for k, objkey in var_assignments.items()
        }
        # {Variable(room): Domain({Semantics.Bathroom, SpecificObject(name='bathroom_0-0'), Semantics.Room}, [])}
        # {Variable(room): Domain({Semantics.Bedroom, SpecificObject(name='bedroom_0-0'), Semantics.Room}, []), Variable(obj): Domain({Semantics.Furniture, FromGenerator(BedFactory), Semantics.Bed, Semantics.Object, Semantics.RealPlaceholder, -Semantics.Room}, [])}

        filter_domain = r.substitute_all(filter_domain, dom_assignments)
        """
        Domain({Semantics.Object, -Semantics.Room}, [
            (StableAgainst({}, {Subpart.SupportSurface, Subpart.Visible, -Subpart.Ceiling, -Subpart.Wall}), Domain({SpecificObject(name='bathroom_0-0'), Semantics.Room, Semantics.Bathroom, -Semantics.Object}, [])),
            (-AnyRelation(), Domain({Semantics.Object, -Semantics.Room}, []))
        ])
        """

        if not r.domain_finalized(filter_domain):
            raise ValueError(
                f"Cannot solve {desc_full=} with non-finalized domain {filter_domain}"
            )

        orig_bounds = r.constraint_bounds(consgraph)  # len(orig_bounds) = 63
        # find objects than can be add to fit requirment
        print_bounds = True
        bounds = propose_discrete.preproc_bounds(
            orig_bounds, self.state, filter_domain, print_bounds=print_bounds
        )
        # #len(bounds) = 5

        if len(bounds) == 0:
            bounds = propose_discrete.preproc_bounds(
                orig_bounds, self.state, filter_domain, print_bounds=print_bounds
            )

            logger.info(f"No objects to be added for {desc_full=}, skipping")
            return self.state

        active_count = greedy.update_active_flags(self.state, var_assignments)  # 5,17

        n_start = len(self.state.objs)  # 37
        logger.info(
            f"Greedily solve {desc_full} - stage has {len(bounds)}/{len(orig_bounds)} bounds, "
            f"{active_count=}/{len(self.state.objs)} objs"
        )
        # [15:25:10.504] [solve] [INFO] | Greedily solve ('on_floor_0', 'bedroom_0-0') - stage has 7/63 bounds, active_count=5/25 objs
        # [10:56:47.439] [solve] [INFO] | Greedily solve ('obj_ontop_obj_0', 'bedroom_0-0', '195935_BedFactory') - stage has 2/63 bounds, active_count=17/37 objs

        self.optim.reset(max_iters=n_steps)
        # ra = (
        #     trange(n_steps) if self.optim.print_report_freq == 0 else range(n_steps)
        # )  # range(0, 150)
        ra = trange(n_steps) if self.optim.print_report_freq == 0 else range(n_steps*2)

        # 进行迭代
        for j in ra:
            # print(j)
            # if j==6:

            move_gen = self.choose_move_type(j, n_steps)  # 选择移动类型
            while move_gen.__name__ != "propose_translate":
                move_gen = self.choose_move_type(j, n_steps)  # 选择移动类型
            # print(move_gen , "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.optim.step(
                consgraph, self.state, move_gen, filter_domain, expand_collision
            )  # MARK # 执行优化步骤
            # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

        self.optim.save_stats(
            self.output_folder / f"optim_{desc}.csv"
        )  # 保存优化统计信息

        logger.info(
            f"Finished solving {desc_full}, added {len(self.state.objs) - n_start} "
            f"objects, loss={self.optim.curr_result.loss():.4f} viol={self.optim.curr_result.viol_count()}"
        )

        logger.info(self.optim.curr_result.to_df())

        violations = {
            k: v for k, v in self.optim.curr_result.violations.items() if v > 0
        }

        if len(violations):
            msg = f"Solver has failed to satisfy constraints for stage {desc_full}. {violations=}."
            if abort_unsatisfied:
                butil.save_blend(self.output_folder / f"abort_{desc}.blend")
                raise ValueError(msg)
            else:
                msg += " Continuing anyway, override `solve_objects.abort_unsatisfied=True` via gin to crash instead."
                logger.warning(msg)

        # re-enable everything so the blender scene populates / displays correctly etc
        for k, v in self.state.objs.items():
            greedy.set_active(self.state, k, True)

        return self.state

    @gin.configurable
    def init_graph(
        self,
        filter_domain: r.Domain,
        var_assignments: dict[str, str],
    ):
        name_mapping = {
                            "Sofa": "seating.SofaFactory",
                            "ArmChair": "seating.ArmChairFactory",
                            "CoffeeTable": "tables.CoffeeTableFactory",
                            "TVStand": "shelves.TVStandFactory",
                            "TV": "appliances.TVFactory",
                            "BookColumn": "table_decorations.BookColumnFactory",
                            "LargeShelf": "shelves.LargeShelfFactory",
                            "FloorLamp": "lamp.FloorLampFactory",
                            "SideTable": "tables.SideTableFactory",
                            "book": None,
                            "remote": None,
                            "magazine": None,
                            "plant": "tableware.PlantContainerFactory",
                            "photo frame": None,
                            "vase": "table_decorations.VaseFactory",
                            "decorative item": None,
                            "coaster": None,
                            "candle": None
                        }
        info_dict = {
                        "Sofa": {"1": {"position": [2.5, 2], "rotation": 0}},
                        "ArmChair": {
                            "1": {"position": [1, 1], "rotation": 0},
                            "2": {"position": [4, 1], "rotation": 0}
                        },
                        "CoffeeTable": {"1": {"position": [2.5, 3.5], "rotation": 0}},
                        "TVStand": {"1": {"position": [2.5, 6.5], "rotation": 270}},
                        "TV": {"1": {"position": [2.5, 6.25], "rotation": 270}},
                        # "BookColumn": {
                        #     "1": {"position": [0.5, 6.5], "rotation": 270},
                        #     "2": {"position": [4.5, 6.5], "rotation": 270}
                        # },
                        "LargeShelf": {"1": {"position": [2.5, 0.5], "rotation": 90}},
                        "FloorLamp": {
                            "1": {"position": [1.5, 2], "rotation": 0},
                            "2": {"position": [3.5, 2], "rotation": 0}
                        },
                        "SideTable": {
                            "1": {"position": [0.75, 1], "rotation": 0},
                            "2": {"position": [4.25, 1], "rotation": 0}
                        }
                    }
        
        dom_assignments = {
            k: r.Domain(self.state.objs[objkey].tags)
            for k, objkey in var_assignments.items()
        }
        filter_domain = r.substitute_all(filter_domain, dom_assignments)

        for key, value in info_dict.items():
            for num in value.keys():
                position = value[num]["position"] + [0]
                rotation = value[num]["rotation"] * math.pi / 180
                name = key
                module_and_class = name_mapping[name]
                module_name, class_name = module_and_class.rsplit('.', 1)
                module = importlib.import_module("infinigen.assets.objects."+module_name)
                class_obj = getattr(module, class_name)
                gen_class = class_obj
                # gen_class = seating.SofaFactory
                search_rels = filter_domain.relations
                # 筛选出有效的关系，只选择非否定关系
                search_rels = [
                    rd for rd in search_rels if not isinstance(rd[0], cl.NegatedRelation)
                ]

                assign = propose_relations.find_assignments(self.state, search_rels)
                for i, assignments in enumerate(assign):
                    found_tags = usage_lookup.usages_of_factory( gen_class )  
                    move = moves.Addition(
                        names=[
                            f"{np.random.randint(1e6):04d}_{gen_class.__name__}"
                        ],  # decided later # 随机生成一个名称，基于生成器类的名称
                        gen_class=gen_class,  # 使用传入的生成器类
                        relation_assignments=assignments,  # 传入分配的关系
                        temp_force_tags=found_tags,  # 临时强制标签
                    )
                    
                    target_name = f"{np.random.randint(1e7)}_{class_name}"
                    # target_name = np.random.randint(1e7)+"_SofaFactory"
                    size = ""
                    
                    meshpath = ""

                    move.apply_init(
                        self.state, target_name, size, position, rotation, gen_class, meshpath
                    )
                    break

        return self.state

    def get_bpy_objects(self, domain: r.Domain) -> list[bpy.types.Object]:
        objkeys = domain_contains.objkeys_in_dom(domain, self.state)
        return [self.state.objs[k].obj for k in objkeys]

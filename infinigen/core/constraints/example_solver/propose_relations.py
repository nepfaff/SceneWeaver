# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import logging
import random
import typing
from pprint import pprint

import numpy as np

from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints import reasoning as r
from infinigen.core.constraints.evaluator.domain_contains import objkeys_in_dom
from infinigen.core.tags import Subpart
import bpy
from . import state_def

logger = logging.getLogger(__name__)


def minimize_redundant_relations(relations: list[tuple[cl.Relation, r.Domain]]):
    """
    Given a list of relations that must be true, use the first as a constraint to tighten the remaining relations
    """

    assert len(relations) > 0

    # TODO Hacky: moves AnyRelations to the back so _hopefully_ they get implied before we get to them
    relations = sorted(
        relations, key=lambda r: isinstance(r[0], cl.AnyRelation), reverse=True
    )

    (rel, dom), *rest = relations

    # Force all remaining relations to be compatible with (rel, dom), thereby reducing their search space
    remaining_relations = []
    for r_later, d_later in rest:
        logger.debug(f"Inspecting {r_later=} {d_later=}")

        if d_later.intersects(dom):
            logger.debug(f"Intersecting {d_later} with {dom}")
            d_later = d_later.intersection(dom)

        if r.reldom_implies((rel, dom), (r_later, d_later)):
            # (rlater, dlater) is guaranteed true so long as we satisfied (rel, dom), we dont need to separately assign it
            logger.debug("Discarding since rlater,dlater it is implied")
            continue
        else:
            logger.debug(
                f"Keeping {r_later, d_later} since it is not implied by {rel, dom} "
            )
            remaining_relations.append((r_later, d_later))

    implied = any(
        r.reldom_implies(reldom_later, (rel, dom))
        for reldom_later in remaining_relations
    )

    return (rel, dom), remaining_relations, implied


def find_assignments(
    curr: state_def.State,
    relations: list[tuple[cl.Relation, r.Domain]],
    assignments: list[state_def.RelationState] = None,
) -> typing.Iterator[list[state_def.RelationState]]:
    """Iterate over possible assignments that satisfy the given relations. Some assignments may not be feasible geometrically -
    a naive implementation of this function would just enumerate all possible objects matching the assignments, and let the solver
    discover that many combinations are impossible. *This* implementation attemps to never generate guaranteed-invalid combinations in the first place.

    Complexity is pretty astronomical:
    - N^M where N is number of candidates per relation, and M is number of relations
    - reduced somewhat when relations intersect or imply eachother
    - luckily, M is typically 1, 2 or 3, as objects arent often related to lots of other objects

    TODO:
    - discover new relations constraints, which can arise from the particular choice of objects
    - prune early when object choice causes bounds to be violated

    This function essentially does a complex form of SAT-solving. It *really* shouldnt be written in python
    """
    """
    遍历可能的分配方案以满足给定的关系约束。注意：
    - 一个简单的实现会枚举所有可能的对象组合，然后让求解器发现哪些组合是不可行的。
    - 本实现尝试避免生成那些几何上必然无效的组合。

    复杂性：
    - N^M：其中 N 是每个关系的候选对象数，M 是关系数。
    - 如果关系之间有交集或彼此隐含，则复杂性会降低。
    - 幸运的是，M 通常是 1、2 或 3，因为对象之间的关系往往不多。

    TODO:
    - 发现因特定对象选择而引入的新关系约束。
    - 当对象选择导致违反边界时提前剪枝。

    这个函数本质上是一个复杂的 SAT（可满足性问题）求解器，理论上不应该用 Python 编写。
    """

    if assignments is None:
        assignments = []
        # print('FIND ASSIGNMENTS TOPLEVEL')
        # pprint(relations)

    if len(relations) == 0:
        yield assignments
        return

    logger.debug(f"Attempting to assign {relations[0]}")  # 调试信息：尝试分配第一个关系

    # 最小化冗余关系：提取当前关系和剩余关系
    (rel, dom), remaining_relations, implied = minimize_redundant_relations(relations)
    assert len(remaining_relations) < len(relations)  # 确保冗余关系被移除

    if implied:  # 如果剩余关系隐含当前关系
        logger.debug(
            f"Found remaining_relations implies {(rel, dom)=}, skipping it"
        )  # 调试信息
        yield from find_assignments(  # 跳过当前关系，继续处理剩余关系
            curr, relations=remaining_relations, assignments=assignments
        )
        return

    if isinstance(rel, cl.AnyRelation):  # 检查是否有未指定的关系
        pprint(relations)
        pprint([(rel, dom)] + remaining_relations)
        raise ValueError(
            f"Got {rel} as first relation. Invalid! Maybe the program is underspecified?"
        )
    # 获取符合约束域的对象候选列表
    candidates = objkeys_in_dom(dom, curr)
    random.shuffle(candidates)

    if len(candidates) == 0:  # YYD add
        yield assignments
        return

    for parent_candidate_name in candidates:  # 遍历候选对象
        if parent_candidate_name != "newroom_0-0":
            a = 1
        logging.debug(f"{parent_candidate_name=}")  # 调试信息
        # 获取当前候选对象的状态
        parent_state = curr.objs[parent_candidate_name]
        # 获取符合关系父标签的平面数量
        n_parent_planes = len(
            curr.planes.get_tagged_planes(parent_state.obj, rel.parent_tags)
        )
        # 随机排列父对象的平面顺序
        parent_order = np.arange(n_parent_planes)
        np.random.shuffle(parent_order)

        for parent_plane in parent_order:  # 遍历每个平面
            # logger.debug(f'Considering {parent_candidate_name=} {parent_plane=} {n_parent_planes=}')
            # 创建一个关系分配实例
            assignment = state_def.RelationState(
                relation=rel,  # 当前关系
                target_name=parent_candidate_name,  # 目标对象
                child_plane_idx=0,  # TODO fill in at apply()-time
                parent_plane_idx=parent_plane,  # 当前父对象的平面索引
            )
            # 递归处理剩余关系
            yield from find_assignments(
                curr,
                relations=remaining_relations,
                assignments=assignments + [assignment],
            )


def find_given_assignments(
    curr: state_def.State,
    relations: list[tuple[cl.Relation, r.Domain]],
    assignments: list[state_def.RelationState] = None,
    parent_obj_name=None,
) -> typing.Iterator[list[state_def.RelationState]]:
    if parent_obj_name=="830203_desk":
        a = 1
    if assignments is None:
        assignments = []
        # print('FIND ASSIGNMENTS TOPLEVEL')
        # pprint(relations)

    if len(relations) == 0:
        yield assignments
        return

    logger.debug(f"Attempting to assign {relations[0]}")  # 调试信息：尝试分配第一个关系

    # 最小化冗余关系：提取当前关系和剩余关系
    (rel, dom), remaining_relations, implied = minimize_redundant_relations(relations)
    assert len(remaining_relations) < len(relations)  # 确保冗余关系被移除

    if implied:  # 如果剩余关系隐含当前关系
        logger.debug(
            f"Found remaining_relations implies {(rel, dom)=}, skipping it"
        )  # 调试信息
        yield from find_assignments(  # 跳过当前关系，继续处理剩余关系
            curr, relations=remaining_relations, assignments=assignments
        )
        return

    # if isinstance(rel, cl.AnyRelation):  # 检查是否有未指定的关系
    #     pprint(relations)
    #     pprint([(rel, dom)] + remaining_relations)
    #     raise ValueError(
    #         f"Got {rel} as first relation. Invalid! Maybe the program is underspecified?"
    #     )
    # 获取符合约束域的对象候选列表
    candidates = objkeys_in_dom(dom, curr)
    random.shuffle(candidates)

    if len(candidates) == 0:  # YYD add
        yield from find_assignments(
            curr,
            relations=remaining_relations,
            assignments=assignments,
        )
        return

    for parent_candidate_name in candidates:  # 遍历候选对象
        if parent_candidate_name != "newroom_0-0":
            parent_candidate_name = parent_obj_name  # when parent is not room, set parent obj name as the given name
        logging.debug(f"{parent_candidate_name=}")  # 调试信息
        # 获取当前候选对象的状态
        parent_state = curr.objs[parent_candidate_name]
        # 获取符合关系父标签的平面数量
        if Subpart.SupportSurface in rel.parent_tags and parent_candidate_name!='newroom_0-0' and hasattr(parent_state, "populate_obj"): #TODO YYD
            populate_obj = bpy.data.objects.get(parent_state.populate_obj)
            n_parent_planes = len(
                curr.planes.get_tagged_planes(populate_obj, rel.parent_tags)
            )
        else:
            n_parent_planes = len(
                curr.planes.get_tagged_planes(parent_state.obj, rel.parent_tags)
            )
        # 随机排列父对象的平面顺序
        parent_order = np.arange(n_parent_planes)
        np.random.shuffle(parent_order)

        for parent_plane in parent_order:  # 遍历每个平面
            # logger.debug(f'Considering {parent_candidate_name=} {parent_plane=} {n_parent_planes=}')
            # 创建一个关系分配实例
            assignment = state_def.RelationState(
                relation=rel,  # 当前关系
                target_name=parent_candidate_name,  # 目标对象
                child_plane_idx=0,  # TODO fill in at apply()-time
                parent_plane_idx=parent_plane,  # 当前父对象的平面索引
            )
            # 递归处理剩余关系
            yield from find_assignments(
                curr,
                relations=remaining_relations,
                assignments=assignments + [assignment],
            )


def find_given_assignments_fast(
    curr: state_def.State,
    relations: list[tuple[cl.Relation, r.Domain]],
    assignments: list[state_def.RelationState] = None,
    parent_obj_name=None,
) -> typing.Iterator[list[state_def.RelationState]]:

    if assignments is None:
        assignments = []
      
    if len(relations) == 0:
        yield assignments
        return

    logger.debug(f"Attempting to assign {relations[0]}")  # 调试信息：尝试分配第一个关系

    # 最小化冗余关系：提取当前关系和剩余关系
    (rel, dom), remaining_relations, implied = minimize_redundant_relations(relations)
    assert len(remaining_relations) < len(relations)  # 确保冗余关系被移除

    if implied:  # 如果剩余关系隐含当前关系
        logger.debug(
            f"Found remaining_relations implies {(rel, dom)=}, skipping it"
        )  # 调试信息
        yield from find_assignments(  # 跳过当前关系，继续处理剩余关系
            curr, relations=remaining_relations, assignments=assignments
        )
        return

    # 获取符合约束域的对象候选列表
    candidates = [parent_obj_name]

    for parent_candidate_name in candidates:  # 遍历候选对象
        if parent_candidate_name != "newroom_0-0":
            parent_candidate_name = parent_obj_name  # when parent is not room, set parent obj name as the given name
        logging.debug(f"{parent_candidate_name=}")  # 调试信息
        # 获取当前候选对象的状态
        parent_state = curr.objs[parent_candidate_name]
        # 获取符合关系父标签的平面数量
        if Subpart.SupportSurface in rel.parent_tags and parent_candidate_name!='newroom_0-0' and hasattr(parent_state, "populate_obj"): #TODO YYD
            populate_obj = bpy.data.objects.get(parent_state.populate_obj)
            n_parent_planes = len(
                curr.planes.get_tagged_planes(populate_obj, rel.parent_tags)
            )
        else:
            n_parent_planes = len(
                curr.planes.get_tagged_planes(parent_state.obj, rel.parent_tags)
            )
        # 随机排列父对象的平面顺序
        parent_order = np.arange(n_parent_planes)
        np.random.shuffle(parent_order)

        for parent_plane in parent_order:  # 遍历每个平面
            # logger.debug(f'Considering {parent_candidate_name=} {parent_plane=} {n_parent_planes=}')
            # 创建一个关系分配实例
            assignment = state_def.RelationState(
                relation=rel,  # 当前关系
                target_name=parent_candidate_name,  # 目标对象
                child_plane_idx=0,  # TODO fill in at apply()-time
                parent_plane_idx=parent_plane,  # 当前父对象的平面索引
            )
            # 递归处理剩余关系
            yield from find_assignments(
                curr,
                relations=[],
                assignments=assignments + [assignment],
            )

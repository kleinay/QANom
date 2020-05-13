from itertools import product
from typing import List, Dict

from qanom.annotations.decode_encode_answers import Argument, Role
from qanom.evaluation.metrics import iou, MATCH_IOU_THRESHOLD


def find_matches(sys_args: List[Argument], grt_args: List[Argument]) -> Dict[Argument, Argument]:
    """ Matching sys and grt arguments based in span overlap (IOU above threshold). """
    matches = ((sys_arg, grt_arg, iou(sys_arg, grt_arg))
               for sys_arg, grt_arg in product(sys_args, grt_args))
    # get any overlapping pairs between SYS and GRT
    matches = [(sys_arg, grt_arg, score)
               for (sys_arg, grt_arg, score) in matches
               if score > MATCH_IOU_THRESHOLD]
    matches = sorted(matches, key=lambda m: m[2], reverse=True)
    if not matches:
        return dict()

    used_sys_args, used_grt_args = set(), set()
    sys_to_grt = {}
    for (sys_arg, grt_arg, score) in matches:
        if grt_arg in used_grt_args or sys_arg in used_sys_args:
            continue
        sys_to_grt[sys_arg] = grt_arg
        used_sys_args.add(sys_arg)
        used_grt_args.add(grt_arg)

    return sys_to_grt


class RoleAlignment:

    def __init__(self, grt_roles: List[Role], sys_roles: List[Role]):
        self.grt_roles = grt_roles
        self.sys_roles = sys_roles
        self.sys_to_grt = {role: set() for role in sys_roles}
        self.grt_to_sys = {role: set() for role in grt_roles}

    def add_alignment(self, grt_role: Role, sys_role: Role):
        self.sys_to_grt[sys_role].add(grt_role)
        self.grt_to_sys[grt_role].add(sys_role)

    def has_single_alignment(self, role: Role, is_grt: bool):
        role_dict = self.grt_to_sys if is_grt else self.sys_to_grt
        n_aligned = len(role_dict[role])
        return n_aligned == 1


def find_role(arg: Argument, roles: List[Role]) -> Role:
    return next(role for role in roles if arg in role.arguments)


def align_by_argument(grt_roles: List[Role], sys_roles: List[Role],
                      sys_to_grt: Dict[Argument, Argument]) -> RoleAlignment:
    alignment = RoleAlignment(grt_roles, sys_roles)
    for sys_role in sys_roles:
        for sys_arg in sys_role.arguments:
            grt_arg = sys_to_grt.get(sys_arg)
            if not grt_arg:
                continue
            grt_role = find_role(grt_arg, grt_roles)
            alignment.add_alignment(grt_role, sys_role)
    return alignment
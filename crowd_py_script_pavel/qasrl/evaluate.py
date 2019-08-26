from dataclasses import dataclass
from itertools import combinations, product
from typing import List, Dict

from common import Role, Argument, Question

MATCH_IOU_THRESHOLD =  0.3


@dataclass
class Metrics:
    true_positive: int
    false_positive: int
    false_negative: int

    def prec(self):
        n_predicted = self.true_positive + self.false_positive
        return float(self.true_positive)/n_predicted

    def recall(self):
        n_true = self.true_positive + self.false_negative
        return float(self.true_positive)/n_true

    def f1(self):
        p, r = self.prec(), self.recall()
        return 2*p*r/(p+r)


def iou(arg1: Argument, arg2: Argument):
    joint = joint_len(arg1, arg2)
    len1 = arg1[1] - arg1[0]
    len2 = arg2[1] - arg2[0]
    union = len1 + len2 - joint
    return float(joint)/union


def joint_len(arg1: Argument, arg2: Argument):
    max_start = max(arg1[0], arg2[0])
    min_end = min(arg1[1], arg2[1])
    joint = max(min_end - max_start, 0)
    return joint


def ensure_no_overlaps(roles: List[Role], is_verbose = False) -> List[Role]:
    # for consistency, and being order independent, remove overlapping arguments
    # in some consistent sorting order.
    roles = sorted(roles, key=lambda role: role.arguments[0][0] + role.arguments[0][1])
    is_done = False
    while not is_done:
        is_done = True
        # for simplicity renew upon every new iteration
        arg_to_role = [(arg, role) for role in roles for arg in role.arguments]
        for (arg_1, role_1), (arg_2, role_2) in combinations(arg_to_role, r=2):
            if joint_len(arg_1, arg_2):
                if is_verbose: 
                    print(f"conflict:\t{arg_1}\t{arg_2}")

                is_done = False
                role_2.remove(arg_2)
                
    roles = [role for role in roles if role.arguments]
    return roles


def find_matches(sys_args: List[Argument], grt_args: List[Argument]) -> Dict[Argument, Argument]:
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


def evaluate(sys_roles: List[Role],
             grt_roles: List[Role],
             allow_overlaps: bool):

    # TODO
    # if not allow_overlaps:
    #     sys_roles = ensure_no_overlaps(sys_roles)

    sys_args = [arg for role in sys_roles for arg in role.arguments]
    grt_args = [arg for role in grt_roles for arg in role.arguments]    
    sys_to_grt = find_matches(sys_args, grt_args)

    arg_metrics = eval_arguments(grt_roles, sys_roles, sys_to_grt)
    role_metrics = eval_roles(grt_roles, sys_roles, sys_to_grt)

    return arg_metrics, role_metrics, sys_to_grt


def count_arguments(roles: List[Role]):
    return sum(len(role.arguments) for role in roles)


def eval_arguments(grt_roles: List[Role], sys_roles: List[Role], sys_to_grt: Dict[Argument, Argument]) -> Metrics:
    tp_arg_count = len(sys_to_grt)
    fp_arg_count = count_arguments(sys_roles) - tp_arg_count
    fn_arg_count = count_arguments(grt_roles) - tp_arg_count
    return Metrics(tp_arg_count, fp_arg_count, fn_arg_count)


class RoleAlignment:

    def __init__(self, grt_roles: List[Role], sys_roles: List[Role]):
        self.grt_roles = grt_roles
        self.grt_roles = grt_roles
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


def align_by_argument(grt_roles: List[Role], sys_roles: List[Role], sys_to_grt: Dict[Argument, Argument]) -> RoleAlignment:
    alignment = RoleAlignment(grt_roles, sys_roles)
    for sys_role in sys_roles:
        for sys_arg in sys_role.arguments:
            grt_arg = sys_to_grt.get(sys_arg)
            if not grt_arg:
                continue
            grt_role = find_role(grt_arg, grt_roles)
            alignment.add_alignment(grt_role, sys_role)
    return alignment


def eval_roles(grt_roles: List[Role],
               sys_roles: List[Role],
               sys_to_grt: Dict[Argument, Argument]) -> Metrics:
    alignemnt = align_by_argument(grt_roles, sys_roles, sys_to_grt)
    tp, fp, fn = 0, 0, 0
    for grt_role in grt_roles: 
        if alignemnt.has_single_alignment(grt_role, is_grt=True):
            tp += 1
        else:
            fn += 1
    for sys_role in sys_roles:
        if not alignemnt.has_single_alignment(sys_role, is_grt=False):
            fp += 1            
    return Metrics(tp, fp, fn)

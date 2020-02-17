from dataclasses import dataclass, astuple
from itertools import combinations, product
from typing import List, Dict

import numpy as np

from annotations.decode_encode_answers import Argument, Role, Response

MATCH_IOU_THRESHOLD = 0.3


@dataclass
class Metrics:
    true_positive: int
    false_positive: int
    false_negative: int

    def prec(self) -> float:
        n_predicted = self.true_positive + self.false_positive
        if not n_predicted:
            return 1.0
        return float(self.true_positive) / n_predicted

    def recall(self) -> float:
        n_true = self.true_positive + self.false_negative
        if not n_true:
            return 1.0
        return float(self.true_positive) / n_true

    def f1(self) -> float:
        p, r = self.prec(), self.recall()
        if p + r == 0:
            return 0
        return 2 * p * r / (p + r)

    def positives(self) -> int:
        return self.true_positive + self.false_positive

    def errors(self) -> int:
        return self.false_negative + self.false_positive

    def accuracy(self, n_decisions: int) -> float:
        return 1 - (self.errors()/float(n_decisions))

    def as_np(self):
        """ Return the tuple representation of this Metrics as numpy array. """
        return np.array(astuple(self))

    """ Operators """
    def __add__(self, other):
        if other == 0:
            return self
        return Metrics( *(self.as_np() + np.array(astuple(other))))

    def __radd__(self, other):
        if other == 0:
            return self
        return Metrics( *(self.as_np() + np.array(astuple(other))))

    def __str__(self):
        return f"P: {self.prec():.3f}   R: {self.recall():.3f}   F1: {self.f1():.3f}"

    def __repr__(self):
        return str(self)

    @classmethod
    def empty(cls):
        return Metrics(0,0,0)


@dataclass
class BinaryClassificationMetrics:
    """ Helper dataclass to compute accuracy on a binary classification task. """
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    def instances(self) -> int:
        """ Number of decisions counted in this Metric """
        return sum([self.true_positive, self.true_negative, self.false_positive, self.false_negative])

    def positives(self) -> int:
        return self.true_positive + self.false_positive

    def errors(self) -> int:
        return self.false_negative + self.false_positive

    def correct_predictions(self) -> int:
        return self.true_positive + self.true_negative

    def accuracy(self) -> float:
        if not self.instances():
            return 1.0
        return self.correct_predictions() / float(self.instances())

    """ When desired, we can also treat it as a retrieval problem with p, r & f1. """
    def prec(self):
        n_predicted = self.true_positive + self.false_positive
        if not n_predicted:
            return 1.0
        return float(self.true_positive) / n_predicted

    def recall(self):
        n_true = self.true_positive + self.false_negative
        if not n_true:
            return 1.0
        return float(self.true_positive) / n_true

    def f1(self):
        p, r = self.prec(), self.recall()
        if p+r == 0:
            return 0
        return 2 * p * r / (p + r)

    def as_np(self):
        """ Return the tuple representation of this Metrics as numpy array. """
        return np.array(astuple(self))

    """ Operators """
    def __add__(self, other):
        if other == 0:
            return self
        return BinaryClassificationMetrics( *(self.as_np() + np.array(astuple(other))))

    def __radd__(self, other):
        if other == 0:
            return self
        return BinaryClassificationMetrics(*(self.as_np() + np.array(astuple(other))))

    def __mul__(self, other: int):
        return BinaryClassificationMetrics( *(self.as_np()*other) )

    def __sub__(self, other):
        return BinaryClassificationMetrics( *(self.as_np() - np.array(astuple(other))))

    @classmethod
    def empty(cls):
        return BinaryClassificationMetrics(0,0,0,0)

    @classmethod
    def simple_boolean_decision(cls, sys_decision: bool, grt_decision: bool):
        tp = int(sys_decision and grt_decision)
        tn = int(not sys_decision and not grt_decision)
        fp = int(sys_decision and not grt_decision)
        fn = int(not sys_decision and grt_decision)
        return BinaryClassificationMetrics(tp,tn,fp,fn)


def iou(arg1: Argument, arg2: Argument):
    joint = joint_len(arg1, arg2)
    len1 = arg1[1] - arg1[0]
    len2 = arg2[1] - arg2[0]
    union = len1 + len2 - joint
    return float(joint) / union


def joint_len(arg1: Argument, arg2: Argument):
    max_start = max(arg1[0], arg2[0])
    min_end = min(arg1[1], arg2[1])
    joint = max(min_end - max_start, 0)
    return joint


def ensure_no_overlaps(roles: List[Role], is_verbose=False) -> List[Role]:
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


def evaluate(sys_response: Response,
             grt_response: Response,
             allow_overlaps: bool):
    # TODO
    # if not allow_overlaps:
    #     sys_roles = ensure_no_overlaps(sys_roles)
    sys_roles: List[Role] = sys_response.roles
    grt_roles: List[Role] = grt_response.roles
    sys_to_grt = find_matches(sys_response.all_args(), grt_response.all_args())

    is_nom_metrics = BinaryClassificationMetrics.simple_boolean_decision(sys_response.is_verbal, grt_response.is_verbal)

    # todo decide on evaluation of roles where is_verbal mismatch - should the roles be included in the role_count metric?
    # Currently excluding these mismatches from the arg & roles metrices
    if is_nom_metrics.errors() == 0:
        arg_metrics = eval_arguments(grt_roles, sys_roles, sys_to_grt)
        role_metrics = eval_roles(grt_roles, sys_roles, sys_to_grt)
    else:
        arg_metrics = Metrics.empty()
        role_metrics = Metrics.empty()

    return arg_metrics, role_metrics, is_nom_metrics, sys_to_grt


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
from ray.rllib.agents import dqn
from ray.rllib.agents.dqn.dqn_policy_graph import DQNPolicyGraph
from ray.tune.registry import register_trainable

from copy import copy


def filter_var_policy_factory(scope_to_freeze=None, scope_not_to_freeze=None):
    """
    Class factory

    Rllib interface expect a class to create the policy_model graph.
    This is used to choose which variable the optimizer optimize.
    (Helps if you want to freeze certain layer for example)

    :param scope_to_freeze:
    :return: a class policy graph
    """
    assert scope_to_freeze is not None or scope_not_to_freeze is not None, "Need to set which variable are frozen or not"
    assert not (scope_to_freeze and scope_not_to_freeze), "Cannot set both freeze and not_to_freeze at the same time"

    class _FilterPolicyGraph(DQNPolicyGraph):
        def __init__(self, *args, **kwargs):

            self.scope_to_freeze = scope_to_freeze
            self.scope_not_to_freeze = scope_not_to_freeze
            super(_FilterPolicyGraph, self).__init__(*args, **kwargs)

        def gradients(self, optimizer):

            saved_vars = copy(self.q_func_vars)

            # var to freeze : indicate which tf scope to freeze
            if self.scope_to_freeze is not None:
                self.q_func_vars = [var for var in self.q_func_vars if self.scope_to_freeze not in var.name]
            #
            else:
                self.q_func_vars = [var for var in self.q_func_vars if self.scope_not_to_freeze in var.name]


            grads_and_vars = super().gradients(optimizer)

            self.q_func_vars = saved_vars
            return grads_and_vars


    return _FilterPolicyGraph

class FilmFrozenApex(dqn.ApexAgent):

    _agent_name = "APEX_FILM_FROZEN"
    _default_config = dqn.apex.APEX_DEFAULT_CONFIG

    _policy_graph = filter_var_policy_factory(scope_to_freeze='film')

class VisionFrozenApex(dqn.ApexAgent):

    _agent_name = "APEX_VISION_FROZEN"
    _default_config = dqn.apex.APEX_DEFAULT_CONFIG

    _policy_graph = filter_var_policy_factory(scope_not_to_freeze='film')


register_trainable("APEX_VISION_FROZEN", VisionFrozenApex)
register_trainable("APEX_FILM_FROZEN", FilmFrozenApex)

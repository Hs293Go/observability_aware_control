from scipy import optimize

from . import observability_cost as oc


class ObservabilityAwareController:
    def __init__(self, cost: oc.ObservabilityCost) -> None:
        self._cost = cost

    def optimize(self, x0, u0, t): ...

from pyvrp._pyvrp import (
    CostEvaluator,
    ProblemData,
    RandomNumberGenerator,
    Solution,
)
from pyvrp.search._search import (
    TabuSearchParams,
    TabuSearchStatistics,
)
from pyvrp.search._search import TabuSearch as _TabuSearch


class TabuSearch:
    """
    Tabu search method with arc promises.

    Parameters
    ----------
    data
        Data object describing the problem to be solved.
    rng
        Random number generator.
    neighbours
        List of lists that defines the search neighbourhood.
    params
        Tabu search parameters.
    """

    def __init__(
        self,
        data: ProblemData,
        rng: RandomNumberGenerator,
        neighbours: list[list[int]],
        params: TabuSearchParams = TabuSearchParams(),
    ):
        self._ts = _TabuSearch(data, neighbours, params)
        self._rng = rng

    @property
    def neighbours(self) -> list[list[int]]:
        """
        Returns the current neighbourhood.
        """
        return self._ts.neighbours

    @neighbours.setter
    def neighbours(self, neighbours: list[list[int]]):
        """
        Replaces the current neighbourhood.
        """
        self._ts.neighbours = neighbours

    @property
    def statistics(self) -> TabuSearchStatistics:
        """
        Returns tabu-search statistics from the last run.
        """
        return self._ts.statistics

    @property
    def promise_cost_tags(self) -> list[list[int]]:
        """
        Returns the promise matrix cost tags.
        """
        return self._ts.promise_cost_tags

    @property
    def promise_excess_demand(self) -> list[list[int]]:
        """
        Returns the promise matrix excess-demand tags.
        """
        return self._ts.promise_excess_demand

    @property
    def promise_time_warp(self) -> list[list[int]]:
        """
        Returns the promise matrix time-warp tags.
        """
        return self._ts.promise_time_warp

    def set_promises(
        self,
        promise_cost_tags: list[list[int]],
        promise_excess_demand: list[list[int]],
        promise_time_warp: list[list[int]],
    ) -> None:
        """
        Sets all promise matrices.
        """
        self._ts.set_promises(
            promise_cost_tags,
            promise_excess_demand,
            promise_time_warp,
        )

    def reset_promises(self) -> None:
        """
        Resets all promise matrices to defaults.
        """
        self._ts.reset_promises()

    def __call__(
        self,
        solution: Solution,
        cost_evaluator: CostEvaluator,
        exhaustive: bool = False,
    ) -> Solution:
        """
        Improves the given solution through tabu search and returns the best
        solution encountered.
        """
        self._ts.shuffle(self._rng)
        return self._ts(solution, cost_evaluator, exhaustive)

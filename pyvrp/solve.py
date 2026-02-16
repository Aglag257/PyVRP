from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING

import pyvrp.search
from pyvrp.IteratedLocalSearch import (
    IteratedLocalSearch,
    IteratedLocalSearchParams,
)
from pyvrp.PenaltyManager import PenaltyManager, PenaltyParams
from pyvrp._pyvrp import ProblemData, RandomNumberGenerator, Solution
from pyvrp.search import (
    OPERATORS,
    BinaryOperator,
    LocalSearch,
    NeighbourhoodParams,
    PerturbationManager,
    PerturbationParams,
    TabuSearch,
    TabuSearchParams,
    UnaryOperator,
    compute_neighbours,
)

if TYPE_CHECKING:
    import pathlib

    from pyvrp.Result import Result
    from pyvrp.stop import StoppingCriterion


class SolveParams:
    """
    Solver parameters for PyVRP's iterated local search algorithm.

    Parameters
    ----------
    ils
        Iterated local search parameters.
    penalty
        Penalty parameters.
    neighbourhood
        Neighbourhood parameters.
    operators
        Operators to use in the search.
    display_interval
        Time (in seconds) between iteration logs. Default 5s.
    perturbation
        Perturbation parameters.
    search_method
        Search method to use inside the ILS loop. One of
        ``\"local_search\"`` (default) or ``\"tabu_search\"``.
    tabu
        Tabu search parameters, used when ``search_method=\"tabu_search\"``.
    """

    def __init__(
        self,
        ils: IteratedLocalSearchParams = IteratedLocalSearchParams(),
        penalty: PenaltyParams = PenaltyParams(),
        neighbourhood: NeighbourhoodParams = NeighbourhoodParams(),
        operators: list[type[UnaryOperator | BinaryOperator]] = OPERATORS,
        display_interval: float = 5.0,
        perturbation: PerturbationParams = PerturbationParams(),
        search_method: str = "local_search",
        tabu: TabuSearchParams = TabuSearchParams(),
    ):
        if search_method not in {"local_search", "tabu_search"}:
            raise ValueError(
                "search_method must be either 'local_search' or "
                "'tabu_search'."
            )

        self._ils = ils
        self._penalty = penalty
        self._neighbourhood = neighbourhood
        self._operators = operators
        self._display_interval = display_interval
        self._perturbation = perturbation
        self._search_method = search_method
        self._tabu = tabu

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SolveParams)
            and self.ils == other.ils
            and self.penalty == other.penalty
            and self.neighbourhood == other.neighbourhood
            and self.operators == other.operators
            and self.display_interval == other.display_interval
            and self.perturbation == other.perturbation
            and self.search_method == other.search_method
            and self.tabu == other.tabu
        )

    @property
    def ils(self):
        return self._ils

    @property
    def penalty(self):
        return self._penalty

    @property
    def neighbourhood(self):
        return self._neighbourhood

    @property
    def operators(self):
        return self._operators

    @property
    def display_interval(self) -> float:
        return self._display_interval

    @property
    def perturbation(self):
        return self._perturbation

    @property
    def search_method(self) -> str:
        return self._search_method

    @property
    def tabu(self):
        return self._tabu

    @classmethod
    def from_file(cls, loc: str | pathlib.Path):
        """
        Loads the solver parameters from a TOML file.
        """
        with open(loc, "rb") as fh:
            data = tomllib.load(fh)

        operators = OPERATORS
        if "operators" in data:
            operators = [getattr(pyvrp.search, op) for op in data["operators"]]

        return cls(
            IteratedLocalSearchParams(**data.get("ils", {})),
            PenaltyParams(**data.get("penalty", {})),
            NeighbourhoodParams(**data.get("neighbourhood", {})),
            operators,
            data.get("display_interval", 5.0),
            PerturbationParams(**data.get("perturbation", {})),
            data.get("search_method", "local_search"),
            TabuSearchParams(**data.get("tabu", {})),
        )


def solve(
    data: ProblemData,
    stop: StoppingCriterion,
    seed: int = 0,
    collect_stats: bool = True,
    display: bool = False,
    params: SolveParams = SolveParams(),
    initial_solution: Solution | None = None,
) -> Result:
    """
    Solves the given problem data instance.

    Parameters
    ----------
    data
        Problem data instance to solve.
    stop
        Stopping criterion to use.
    seed
        Seed value to use for the random number stream. Default 0.
    collect_stats
        Whether to collect statistics about the solver's progress. Default
        ``True``.
    display
        Whether to display information about the solver progress. Default
        ``False``. Progress information is only available when
        ``collect_stats`` is also set, which it is by default.
    params
        Solver parameters to use. If not provided, a default will be used.
    initial_solution
        Optional solution to use as a warm start. The solver constructs a
        (possibly poor) initial solution if this argument is not provided.

    Returns
    -------
    Result
        A Result object, containing statistics (if collected) and the best
        found solution.
    """
    rng = RandomNumberGenerator(seed=seed)
    neighbours = compute_neighbours(data, params.neighbourhood)
    if params.search_method == "local_search":
        perturbation = PerturbationManager(params.perturbation)
        search = LocalSearch(data, rng, neighbours, perturbation)

        for op in params.operators:
            if op.supports(data):
                search.add_operator(op(data))
    else:
        search = TabuSearch(data, rng, neighbours, params.tabu)

    pm = PenaltyManager.init_from(data, params.penalty)

    init = initial_solution
    if init is None:
        # Start from a random initial solution to ensure it's not completely
        # empty (because starting from empty solutions can be a bit difficult).
        random = Solution.make_random(data, rng)
        init = search(random, pm.max_cost_evaluator(), exhaustive=True)

    algo = IteratedLocalSearch(data, pm, rng, search, init, params.ils)
    return algo.run(stop, collect_stats, display, params.display_interval)

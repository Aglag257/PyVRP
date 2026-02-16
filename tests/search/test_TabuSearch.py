from numpy.testing import assert_, assert_equal

from pyvrp import CostEvaluator, RandomNumberGenerator, Solution
from pyvrp.search import TabuSearch, TabuSearchParams, compute_neighbours


def test_tabu_search_returns_best_encountered_solution(ok_small):
    """
    Tests that tabu search returns a solution that is no worse than the
    starting solution under the used penalised objective.
    """
    rng = RandomNumberGenerator(seed=42)
    neighbours = compute_neighbours(ok_small)

    tabu = TabuSearch(
        ok_small,
        rng,
        neighbours,
        TabuSearchParams(max_iterations=50, tabu_tenure=5),
    )

    cost_evaluator = CostEvaluator([20], 6, 0)
    sol = Solution.make_random(ok_small, rng)

    improved = tabu(sol, cost_evaluator, exhaustive=True)

    assert_(
        cost_evaluator.penalised_cost(improved)
        <= cost_evaluator.penalised_cost(sol)
    )
    assert_(tabu.statistics.num_moves >= 0)


def test_tabu_search_promise_matrix_roundtrip(ok_small):
    """
    Tests that promise matrices can be set and read back unchanged.
    """
    rng = RandomNumberGenerator(seed=42)
    num_locs = ok_small.num_locations

    tabu = TabuSearch(
        ok_small,
        rng,
        compute_neighbours(ok_small),
        TabuSearchParams(max_iterations=5, tabu_tenure=2),
    )

    cost_tags = [
        [10_000 + i + j for j in range(num_locs)] for i in range(num_locs)
    ]
    excess_tags = [
        [100 + i + j for j in range(num_locs)] for i in range(num_locs)
    ]
    time_warp_tags = [
        [200 + i + j for j in range(num_locs)] for i in range(num_locs)
    ]

    tabu.set_promises(cost_tags, excess_tags, time_warp_tags)

    assert_equal(tabu.promise_cost_tags, cost_tags)
    assert_equal(tabu.promise_excess_demand, excess_tags)
    assert_equal(tabu.promise_time_warp, time_warp_tags)

    tabu.reset_promises()

    assert_(tabu.promise_cost_tags[0][1] > cost_tags[0][1])
    assert_(tabu.promise_excess_demand[0][1] > excess_tags[0][1])
    assert_(tabu.promise_time_warp[0][1] > time_warp_tags[0][1])

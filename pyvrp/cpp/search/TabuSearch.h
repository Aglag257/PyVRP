#ifndef PYVRP_SEARCH_TABUSEARCH_H
#define PYVRP_SEARCH_TABUSEARCH_H

#include "Exchange.h"
#include "ProblemData.h"
#include "RandomNumberGenerator.h"
#include "SearchSpace.h"
#include "Solution.h"  // pyvrp::search::Solution

#include <cstdint>
#include <utility>
#include <vector>

namespace pyvrp::search
{
/**
 * Parameters for tabu search with arc promises.
 *
 * Attributes
 * ----------
 * max_iterations
 *     Maximum number of tabu-search iterations.
 * tabu_tenure
 *     Number of iterations for which a tabu move remains forbidden.
 * reset_frequency
 *     Number of iterations between global promise-tag resets. ``0`` disables
 *     periodic resets.
 * aspiration
 *     Allows tabu moves when they improve over the best known solution.
 * reset_on_best
 *     Resets all promise tags when a new incumbent best solution is found.
 * consider_empty_routes
 *     Whether to also evaluate relocate moves into empty routes.
 */
struct TabuSearchParams
{
    size_t const maxIterations;
    size_t const tabuTenure;
    size_t const resetFrequency;
    bool const aspiration;
    bool const resetOnBest;
    bool const considerEmptyRoutes;

    TabuSearchParams(size_t maxIterations = 250,
                     size_t tabuTenure = 20,
                     size_t resetFrequency = 0,
                     bool aspiration = true,
                     bool resetOnBest = true,
                     bool considerEmptyRoutes = true);

    bool operator==(TabuSearchParams const &other) const = default;
};

/**
 * Arc-based tabu search with a promise-tag matrix.
 *
 * Promise tags are associated with arcs. A move is promise-keeping only when
 * all arcs it introduces satisfy the aspiration-like rule:
 *
 *    candidate_cost < promise_cost_tag[arc]
 *
 * Besides the cost tags, this implementation also stores per-arc excess-demand
 * and time-warp tags so external penalty controllers can read/write those
 * values.
 */
class TabuSearch
{
public:
    using Matrix = std::vector<std::vector<int64_t>>;

    /**
     * Statistics collected during the most recent search run.
     *
     * Attributes
     * ----------
     * num_iterations
     *     Number of completed tabu-search iterations.
     * num_moves
     *     Number of applied moves.
     * num_aspiration_moves
     *     Number of applied moves accepted through aspiration.
     * num_tabu_rejected
     *     Number of candidate moves rejected by tabu tenure.
     * num_promise_rejected
     *     Number of candidate moves rejected by promise checks.
     */
    struct Statistics
    {
        size_t const numIterations;
        size_t const numMoves;
        size_t const numAspirationMoves;
        size_t const numTabuRejected;
        size_t const numPromiseRejected;
    };

private:
    ProblemData const &data_;
    Solution solution_;
    SearchSpace searchSpace_;
    Exchange<1, 0> relocate_;
    TabuSearchParams params_;

    std::vector<std::vector<size_t>> tabuUntil_;

    Matrix promiseCostTags_;
    Matrix promiseExcessDemand_;
    Matrix promiseTimeWarp_;

    size_t numIterations_ = 0;
    size_t numMoves_ = 0;
    size_t numAspirationMoves_ = 0;
    size_t numTabuRejected_ = 0;
    size_t numPromiseRejected_ = 0;

    void validateMatrix(Matrix const &matrix) const;

    bool isTabu(size_t uClient, size_t vClient, size_t iteration) const;

    bool isPromiseKeeping(Route::Node const *U,
                          Route::Node const *V,
                          Cost candidateCost) const;

    void setArcPromise(size_t from,
                       size_t to,
                       int64_t costTag,
                       int64_t excessDemandTag,
                       int64_t timeWarpTag);

    void resetPromiseMatrices();

    static int64_t routeExcessDemand(Route const *route);
    static int64_t routeTimeWarp(Route const *route);

    std::pair<int64_t, int64_t> routeSignals(Route const *uRoute,
                                              Route const *vRoute) const;

public:
    TabuSearch(ProblemData const &data,
               SearchSpace::Neighbours neighbours,
               TabuSearchParams params = TabuSearchParams());

    /**
     * Set neighbourhood structure to use by tabu search.
     */
    void setNeighbours(SearchSpace::Neighbours neighbours);

    /**
     * Returns the current neighbourhood structure.
     */
    SearchSpace::Neighbours const &neighbours() const;

    /**
     * Returns search statistics for the most recent run.
     */
    Statistics statistics() const;

    /**
     * Returns the current promise cost tags.
     */
    Matrix const &promiseCostTags() const;

    /**
     * Returns the current promise excess-demand tags.
     */
    Matrix const &promiseExcessDemand() const;

    /**
     * Returns the current promise time-warp tags.
     */
    Matrix const &promiseTimeWarp() const;

    /**
     * Sets all promise matrices from user-provided values.
     */
    void setPromises(Matrix promiseCostTags,
                     Matrix promiseExcessDemand,
                     Matrix promiseTimeWarp);

    /**
     * Resets all promise matrices to permissive defaults.
     */
    void resetPromises();

    /**
     * Performs tabu search around the given solution and returns the best
     * encountered solution.
     */
    pyvrp::Solution operator()(pyvrp::Solution const &solution,
                               CostEvaluator const &costEvaluator,
                               bool exhaustive = false);

    /**
     * Shuffles search order.
     */
    void shuffle(RandomNumberGenerator &rng);
};
}  // namespace pyvrp::search

#endif  // PYVRP_SEARCH_TABUSEARCH_H

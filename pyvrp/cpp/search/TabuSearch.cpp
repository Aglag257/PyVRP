#include "TabuSearch.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <limits>
#include <optional>
#include <stdexcept>

using pyvrp::search::Route;
using pyvrp::search::SearchSpace;
using pyvrp::search::TabuSearch;
using pyvrp::search::TabuSearchParams;

TabuSearchParams::TabuSearchParams(size_t maxIterations,
                                   size_t tabuTenure,
                                   size_t resetFrequency,
                                   bool aspiration,
                                   bool resetOnBest,
                                   bool considerEmptyRoutes)
    : maxIterations(maxIterations),
      tabuTenure(tabuTenure),
      resetFrequency(resetFrequency),
      aspiration(aspiration),
      resetOnBest(resetOnBest),
      considerEmptyRoutes(considerEmptyRoutes)
{
    if (maxIterations == 0)
        throw std::runtime_error("max_iterations must be positive.");
}

TabuSearch::TabuSearch(ProblemData const &data,
                       SearchSpace::Neighbours neighbours,
                       TabuSearchParams params)
    : data_(data),
      solution_(data),
      searchSpace_(data, std::move(neighbours)),
      relocate_(data),
      params_(params),
      tabuUntil_(data.numLocations(), std::vector<size_t>(data.numLocations())),
      promiseCostTags_(data.numLocations(),
                       std::vector<int64_t>(data.numLocations())),
      promiseExcessDemand_(data.numLocations(),
                           std::vector<int64_t>(data.numLocations())),
      promiseTimeWarp_(data.numLocations(),
                       std::vector<int64_t>(data.numLocations()))
{
    resetPromiseMatrices();
}

void TabuSearch::validateMatrix(Matrix const &matrix) const
{
    if (matrix.size() != data_.numLocations())
        throw std::runtime_error("Promise matrix dimensions do not match.");

    for (auto const &row : matrix)
        if (row.size() != data_.numLocations())
            throw std::runtime_error("Promise matrix dimensions do not match.");
}

bool TabuSearch::isTabu(size_t uClient, size_t vClient, size_t iteration) const
{
    assert(uClient < data_.numLocations());
    assert(vClient < data_.numLocations());
    return iteration < tabuUntil_[uClient][vClient];
}

bool TabuSearch::isPromiseKeeping(Route::Node const *U,
                                  Route::Node const *V,
                                  Cost candidateCost) const
{
    assert(U->route());
    assert(V->route());

    auto const pU = p(U)->client();
    auto const nU = n(U)->client();
    auto const nV = n(V)->client();

    auto const candidate = candidateCost.get();
    std::array<std::pair<size_t, size_t>, 3> const introduced{
        std::pair<size_t, size_t>{pU, nU},
        std::pair<size_t, size_t>{V->client(), U->client()},
        std::pair<size_t, size_t>{U->client(), nV},
    };

    for (auto const &[from, to] : introduced)
        if (candidate >= promiseCostTags_[from][to])
            return false;

    return true;
}

void TabuSearch::setArcPromise(size_t from,
                               size_t to,
                               int64_t costTag,
                               int64_t excessDemandTag,
                               int64_t timeWarpTag)
{
    assert(from < data_.numLocations());
    assert(to < data_.numLocations());

    promiseCostTags_[from][to] = costTag;
    promiseCostTags_[to][from] = costTag;

    promiseExcessDemand_[from][to] = excessDemandTag;
    promiseExcessDemand_[to][from] = excessDemandTag;

    promiseTimeWarp_[from][to] = timeWarpTag;
    promiseTimeWarp_[to][from] = timeWarpTag;
}

void TabuSearch::resetPromiseMatrices()
{
    auto constexpr maxTag = std::numeric_limits<int64_t>::max();

    for (auto &row : promiseCostTags_)
        std::fill(row.begin(), row.end(), maxTag);

    for (auto &row : promiseExcessDemand_)
        std::fill(row.begin(), row.end(), maxTag);

    for (auto &row : promiseTimeWarp_)
        std::fill(row.begin(), row.end(), maxTag);
}

int64_t TabuSearch::routeExcessDemand(Route const *route)
{
    if (!route)
        return 0;

    int64_t excess = 0;
    for (auto const load : route->excessLoad())
        excess += load.get();

    return excess;
}

int64_t TabuSearch::routeTimeWarp(Route const *route)
{
    return route ? route->timeWarp().get() : 0;
}

std::pair<int64_t, int64_t> TabuSearch::routeSignals(Route const *uRoute,
                                                      Route const *vRoute) const
{
    auto excess = routeExcessDemand(uRoute);
    auto timeWarp = routeTimeWarp(uRoute);

    if (vRoute && vRoute != uRoute)
    {
        excess += routeExcessDemand(vRoute);
        timeWarp += routeTimeWarp(vRoute);
    }

    return {excess, timeWarp};
}

void TabuSearch::setNeighbours(SearchSpace::Neighbours neighbours)
{
    searchSpace_.setNeighbours(neighbours);
}

SearchSpace::Neighbours const &TabuSearch::neighbours() const
{
    return searchSpace_.neighbours();
}

TabuSearch::Statistics TabuSearch::statistics() const
{
    return {
        numIterations_,
        numMoves_,
        numAspirationMoves_,
        numTabuRejected_,
        numPromiseRejected_,
    };
}

TabuSearch::Matrix const &TabuSearch::promiseCostTags() const
{
    return promiseCostTags_;
}

TabuSearch::Matrix const &TabuSearch::promiseExcessDemand() const
{
    return promiseExcessDemand_;
}

TabuSearch::Matrix const &TabuSearch::promiseTimeWarp() const
{
    return promiseTimeWarp_;
}

void TabuSearch::setPromises(Matrix promiseCostTags,
                             Matrix promiseExcessDemand,
                             Matrix promiseTimeWarp)
{
    validateMatrix(promiseCostTags);
    validateMatrix(promiseExcessDemand);
    validateMatrix(promiseTimeWarp);

    promiseCostTags_ = std::move(promiseCostTags);
    promiseExcessDemand_ = std::move(promiseExcessDemand);
    promiseTimeWarp_ = std::move(promiseTimeWarp);
}

void TabuSearch::resetPromises() { resetPromiseMatrices(); }

pyvrp::Solution TabuSearch::operator()(pyvrp::Solution const &solution,
                                       CostEvaluator const &costEvaluator,
                                       bool exhaustive)
{
    struct CandidateMove
    {
        Route::Node *U = nullptr;
        Route::Node *V = nullptr;
        size_t uClient = 0;
        size_t vClient = 0;
        Cost deltaCost = 0;
        bool aspiration = false;
    };

    solution_.load(solution);

    numIterations_ = 0;
    numMoves_ = 0;
    numAspirationMoves_ = 0;
    numTabuRejected_ = 0;
    numPromiseRejected_ = 0;

    for (auto &row : tabuUntil_)
        std::fill(row.begin(), row.end(), 0);

    searchSpace_.markAllPromising();

    std::optional<pyvrp::Solution> bestSolution;
    bestSolution.emplace(solution_.unload());
    auto currentCost = costEvaluator.penalisedCost(solution_);
    auto bestCost = currentCost;

    for (size_t iter = 0; iter != params_.maxIterations; ++iter)
    {
        if (params_.resetFrequency != 0 && iter != 0
            && iter % params_.resetFrequency == 0)
            resetPromiseMatrices();

        bool hasMove = false;
        CandidateMove bestMove;

        auto consider = [&](Route::Node *U, Route::Node *V)
        {
            assert(U);
            assert(V);

            auto const [deltaCost, improving] = relocate_.evaluate(
                U, V, costEvaluator);

            // Invalid relocate moves return (0, false).
            if (!improving && deltaCost == 0)
                return;

            auto const candidateCost = currentCost + deltaCost;
            auto const aspiration = params_.aspiration
                                    && candidateCost < bestCost;

            if (isTabu(U->client(), V->client(), iter) && !aspiration)
            {
                numTabuRejected_++;
                return;
            }

            if (!isPromiseKeeping(U, V, candidateCost))
            {
                numPromiseRejected_++;
                return;
            }

            if (!hasMove || deltaCost < bestMove.deltaCost
                || (deltaCost == bestMove.deltaCost
                    && aspiration && !bestMove.aspiration))
            {
                hasMove = true;
                bestMove = {
                    U,
                    V,
                    U->client(),
                    V->client(),
                    deltaCost,
                    aspiration,
                };
            }
        };

        for (auto const uClient : searchSpace_.clientOrder())
        {
            auto *U = &solution_.nodes[uClient];
            if (!U->route())
                continue;

            if (!exhaustive && !searchSpace_.isPromising(uClient))
                continue;

            for (auto const vClient : searchSpace_.neighboursOf(uClient))
            {
                auto *V = &solution_.nodes[vClient];
                if (!V->route())
                    continue;

                consider(U, V);
            }

            if (!params_.considerEmptyRoutes)
                continue;

            for (auto const &[vehType, offset] : searchSpace_.vehTypeOrder())
            {
                auto const begin = solution_.routes.begin() + offset;
                auto const end
                    = begin + data_.vehicleType(vehType).numAvailable;
                auto const pred = [](auto const &route) { return route.empty(); };
                auto empty = std::find_if(begin, end, pred);

                if (empty != end)
                    consider(U, (*empty)[0]);
            }
        }

        if (!hasMove)
            break;

        auto *U = bestMove.U;
        auto *V = bestMove.V;

        auto *uRoute = U->route();
        auto *vRoute = V->route();
        assert(uRoute && vRoute);

        auto const pU = p(U)->client();
        auto const nU = n(U)->client();
        auto const nV = n(V)->client();

        auto const costTag = currentCost.get();
        auto const [excessTag, timeWarpTag] = routeSignals(uRoute, vRoute);

        relocate_.apply(U, V);

        uRoute->update();
        if (vRoute != uRoute)
            vRoute->update();

        currentCost = costEvaluator.penalisedCost(solution_);

        // Mark moved node and its local neighbourhood as promising.
        searchSpace_.markPromising(U);
        searchSpace_.markPromising(V);

        tabuUntil_[bestMove.uClient][bestMove.vClient]
            = iter + params_.tabuTenure;
        tabuUntil_[bestMove.uClient][pU] = iter + params_.tabuTenure;

        // Tag eliminated arcs with the cost and route-violation signatures of
        // the incumbent before applying the accepted move.
        setArcPromise(pU, bestMove.uClient, costTag, excessTag, timeWarpTag);
        setArcPromise(bestMove.uClient, nU, costTag, excessTag, timeWarpTag);
        setArcPromise(bestMove.vClient, nV, costTag, excessTag, timeWarpTag);

        numIterations_ = iter + 1;
        numMoves_++;
        if (bestMove.aspiration)
            numAspirationMoves_++;

        if (currentCost < bestCost)
        {
            bestCost = currentCost;
            bestSolution.emplace(solution_.unload());

            if (params_.resetOnBest)
                resetPromiseMatrices();
        }
    }

    return std::move(bestSolution).value();
}

void TabuSearch::shuffle(RandomNumberGenerator &rng)
{
    searchSpace_.shuffle(rng);
}

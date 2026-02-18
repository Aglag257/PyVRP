#ifndef PYVRP_REPAIR_H
#define PYVRP_REPAIR_H

#include "CostEvaluator.h"
#include "ProblemData.h"
#include "Solution.h"
#include "search_legacy/Route.h"

#include <vector>

namespace pyvrp::repair
{
// Populate the given locs and routes vectors with routes from the solution.
void setupRoutes(std::vector<search_legacy::Route::Node> &locs,
                 std::vector<search_legacy::Route> &routes,
                 std::vector<pyvrp::Route> const &solRoutes,
                 ProblemData const &data);

// Turns the given search routes into solution routes.
std::vector<pyvrp::Route>
exportRoutes(ProblemData const &data,
             std::vector<search_legacy::Route> const &routes);
}  // namespace pyvrp::repair

#endif  // PYVRP_REPAIR_H

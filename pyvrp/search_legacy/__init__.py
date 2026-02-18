from typing import Type

from .LocalSearch import LocalSearch as LocalSearch
from .SearchMethod import SearchMethod as SearchMethod
from ._search_legacy import Exchange10 as Exchange10
from ._search_legacy import Exchange11 as Exchange11
from ._search_legacy import Exchange20 as Exchange20
from ._search_legacy import Exchange21 as Exchange21
from ._search_legacy import Exchange22 as Exchange22
from ._search_legacy import Exchange30 as Exchange30
from ._search_legacy import Exchange31 as Exchange31
from ._search_legacy import Exchange32 as Exchange32
from ._search_legacy import Exchange33 as Exchange33
from ._search_legacy import NodeOperator as NodeOperator
from ._search_legacy import RelocateWithDepot as RelocateWithDepot
from ._search_legacy import RouteOperator as RouteOperator
from ._search_legacy import SwapRoutes as SwapRoutes
from ._search_legacy import SwapStar as SwapStar
from ._search_legacy import SwapTails as SwapTails
from pyvrp.search.neighbourhood import NeighbourhoodParams as NeighbourhoodParams
from pyvrp.search.neighbourhood import compute_neighbours as compute_neighbours

NODE_OPERATORS: list[Type[NodeOperator]] = [
    Exchange10,
    Exchange20,
    Exchange11,
    Exchange21,
    Exchange22,
    SwapTails,
    RelocateWithDepot,
]

ROUTE_OPERATORS: list[Type[RouteOperator]] = [
    SwapRoutes,
    SwapStar,
]

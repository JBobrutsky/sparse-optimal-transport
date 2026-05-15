# Bonneel Network Simplex

Source: https://github.com/nbonneel/network_simplex  
Files:
- `network_simplex_simple.h` — core network simplex algorithm
- `full_bipartitegraph.h` — required bipartite graph structure (included by network_simplex_simple.h)

License: See header files (LEMON library license with modifications by Nicolas Bonneel)  
Vendored: 2026-05-15  
Modifications: none

## API Reference

The header file implements the **NetworkSimplexSimple** template class for solving minimum cost flow problems using the primal Network Simplex algorithm.

### Template Parameters
```cpp
template <typename GR, typename V = int, typename C = V, typename ArcsType = int64_t>
class NetworkSimplexSimple
```

- `GR`: The digraph type (e.g., full bipartite graph, LEMON graph)
- `V`: Value type for flow amounts (default: int)
- `C`: Cost type for edge costs (default: same as V)
- `ArcsType`: Arc identifier type (default: int64_t)

### Constructor
```cpp
NetworkSimplexSimple(const GR& graph, bool arc_mixing, int nbnodes, ArcsType nb_arcs, size_t maxiters = 0)
```

### Type Definitions
```cpp
typedef V Value;      // Flow amounts
typedef C Cost;       // Arc costs
```

### Problem Types (Enum)
```cpp
enum ProblemType {
    INFEASIBLE,  // No feasible solution
    OPTIMAL,     // Feasible and bounded with optimal solution found
    UNBOUNDED    // Objective function is unbounded
};

enum SupplyType {
    GEQ,  // Greater-or-equal supply constraints (default)
    LEQ   // Less-or-equal supply constraints
};
```

### Key Methods

#### Setting Problem Parameters
- `setCost(const Arc& arc, const Value cost)` → `NetworkSimplexSimple&` - Set cost for a single arc
- `costMap(const CostMap& map)` → `NetworkSimplexSimple&` - Set costs for all arcs
- `supplyMap(const SupplyMap& map)` → `NetworkSimplexSimple&` - Set node supply/demand values
- `stSupply(const Node& s, const Node& t, Value k)` → `NetworkSimplexSimple&` - Set single source-sink pair with flow amount
- `supplyType(SupplyType supply_type)` → `NetworkSimplexSimple&` - Set supply constraint type

#### Running the Algorithm
- `run()` → `ProblemType` - Execute the algorithm and return problem type

#### Retrieving Results
- `flow(const Arc& a)` → `Value` - Get the flow value on an arc
- `flowMap(FlowMap &map)` - Copy all arc flows into a map
- `potential(const Node& n)` → `Cost` - Get the dual potential of a node
- `potentialMap(PotentialMap &map)` - Copy all node potentials into a map

### Notes

- The class uses **sparse flow** representation by default (controlled by `SPARSE_FLOW` macro), which is 10-15% slower for small problems but more memory-efficient for large ones
- Includes OpenMP parallelization in the pivot rule search (March 2015 revision)
- Uses 64-bit integers by default to reduce overflow risks
- The Arc and Node types are defined by the digraph template parameter (GR) via `TEMPLATE_DIGRAPH_TYPEDEFS(GR)` macro
- This is a lightweight implementation adapted from LEMON by Nicolas Bonneel, designed specifically for mass transport problems

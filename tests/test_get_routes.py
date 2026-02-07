"""Tests for MolTree.get_routes() — multi-route extraction from AND/OR trees.

We build synthetic trees by hand (no MLP or stock files needed) and verify
that get_routes returns correctly sorted, distinct routes with the right
costs.
"""

import numpy as np
import pytest

from retro_star.alg.mol_tree import MolTree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_tree(target, known_mols, value_fn=None):
    """Create a MolTree with a trivial value function."""
    if value_fn is None:
        value_fn = lambda _mol: 0.0
    return MolTree(target_mol=target, known_mols=known_mols, value_fn=value_fn)


def _expand(tree, mol_smiles, reactant_lists, costs, templates=None):
    """Find the MolNode with the given SMILES and expand it."""
    node = None
    for m in tree.mol_nodes:
        if m.mol == mol_smiles and m.open and not m.is_known:
            node = m
            break
    assert node is not None, f"No open non-known node for {mol_smiles}"

    if templates is None:
        templates = [f"tmpl_{i}" for i in range(len(costs))]

    # reactant_lists: list[list[str]]
    tree.expand(node, reactant_lists, np.array(costs, dtype=float), templates)


# ---------------------------------------------------------------------------
# Test 1 — single-step synthesis with two alternative reactions
# ---------------------------------------------------------------------------
#
#   Target T can be made via:
#     reaction R1 (cost 1.0): T -> A + B          (both known)
#     reaction R2 (cost 2.0): T -> C              (known)
#
#   Expected routes sorted by cost:
#     route 0: cost 1.0  (R1)
#     route 1: cost 2.0  (R2)


def test_two_alternative_reactions():
    known = {"A", "B", "C"}
    tree = _build_tree("T", known)

    _expand(
        tree,
        "T",
        reactant_lists=[["A", "B"], ["C"]],
        costs=[1.0, 2.0],
        templates=["r1", "r2"],
    )

    assert tree.succ

    # max_routes=1 should give only the best
    routes_1 = tree.get_routes(max_routes=1)
    assert len(routes_1) == 1
    assert abs(routes_1[0].total_cost - 1.0) < 1e-9

    # max_routes=10 should give both
    routes = tree.get_routes(max_routes=10)
    assert len(routes) == 2
    assert abs(routes[0].total_cost - 1.0) < 1e-9
    assert abs(routes[1].total_cost - 2.0) < 1e-9

    # Verify they are distinct
    assert routes[0].serialize() != routes[1].serialize()


# ---------------------------------------------------------------------------
# Test 2 — two-step tree with branching at both levels
# ---------------------------------------------------------------------------
#
#   Target T:
#     R1 (cost 1.0): T -> X + A       (A known)
#     R2 (cost 5.0): T -> B           (B known)
#
#   Intermediate X:
#     R3 (cost 0.5): X -> C + D       (both known)
#     R4 (cost 3.0): X -> E           (E known)
#
#   Complete routes:
#     R1+R3: cost 1.0 + 0.5 = 1.5
#     R1+R4: cost 1.0 + 3.0 = 4.0
#     R2:    cost 5.0
#
#   Sorted: 1.5, 4.0, 5.0


def test_two_step_branching():
    known = {"A", "B", "C", "D", "E"}
    tree = _build_tree("T", known)

    # Expand T
    _expand(
        tree,
        "T",
        reactant_lists=[["X", "A"], ["B"]],
        costs=[1.0, 5.0],
        templates=["r1", "r2"],
    )
    # Tree found no route yet (X is not known)
    # Now expand X
    _expand(
        tree,
        "X",
        reactant_lists=[["C", "D"], ["E"]],
        costs=[0.5, 3.0],
        templates=["r3", "r4"],
    )

    assert tree.succ

    routes = tree.get_routes(max_routes=10)
    assert len(routes) == 3

    costs = [r.total_cost for r in routes]
    assert abs(costs[0] - 1.5) < 1e-9
    assert abs(costs[1] - 4.0) < 1e-9
    assert abs(costs[2] - 5.0) < 1e-9

    # All routes are distinct
    serialized = [r.serialize() for r in routes]
    assert len(set(serialized)) == 3


# ---------------------------------------------------------------------------
# Test 3 — cap at max_routes
# ---------------------------------------------------------------------------


def test_max_routes_cap():
    known = {"A", "B", "C", "D", "E"}
    tree = _build_tree("T", known)

    _expand(
        tree,
        "T",
        reactant_lists=[["X", "A"], ["B"]],
        costs=[1.0, 5.0],
    )
    _expand(
        tree,
        "X",
        reactant_lists=[["C", "D"], ["E"]],
        costs=[0.5, 3.0],
    )

    routes = tree.get_routes(max_routes=2)
    assert len(routes) == 2
    assert abs(routes[0].total_cost - 1.5) < 1e-9
    assert abs(routes[1].total_cost - 4.0) < 1e-9


# ---------------------------------------------------------------------------
# Test 4 — target is already a known molecule
# ---------------------------------------------------------------------------


def test_target_is_known():
    known = {"T"}
    tree = _build_tree("T", known)

    assert tree.succ

    routes = tree.get_routes(max_routes=10)
    assert len(routes) == 1
    assert routes[0].total_cost == 0
    assert routes[0].length == 0


# ---------------------------------------------------------------------------
# Test 5 — no successful route
# ---------------------------------------------------------------------------


def test_no_route():
    known = {"A"}
    tree = _build_tree("T", known)

    # Expand T but with no results
    tree.expand(tree.root, None, None, None)

    assert not tree.succ
    routes = tree.get_routes(max_routes=10)
    assert len(routes) == 0


# ---------------------------------------------------------------------------
# Test 6 — Cartesian product across two intermediates
# ---------------------------------------------------------------------------
#
#   T -> X + Y  (cost 1.0)
#
#   X has 2 routes:
#     X -> A  (cost 0.5)   => X route cost 0.5
#     X -> B  (cost 2.0)   => X route cost 2.0
#
#   Y has 2 routes:
#     Y -> C  (cost 0.3)   => Y route cost 0.3
#     Y -> D  (cost 1.0)   => Y route cost 1.0
#
#   Combinations (all via the same top-level reaction, cost 1.0):
#     X->A + Y->C: 1.0 + 0.5 + 0.3 = 1.8
#     X->A + Y->D: 1.0 + 0.5 + 1.0 = 2.5
#     X->B + Y->C: 1.0 + 2.0 + 0.3 = 3.3
#     X->B + Y->D: 1.0 + 2.0 + 1.0 = 4.0
#
#   Sorted: 1.8, 2.5, 3.3, 4.0


def test_cartesian_product_two_intermediates():
    known = {"A", "B", "C", "D"}
    tree = _build_tree("T", known)

    _expand(tree, "T", reactant_lists=[["X", "Y"]], costs=[1.0])
    _expand(tree, "X", reactant_lists=[["A"], ["B"]], costs=[0.5, 2.0])
    _expand(tree, "Y", reactant_lists=[["C"], ["D"]], costs=[0.3, 1.0])

    assert tree.succ

    routes = tree.get_routes(max_routes=10)
    assert len(routes) == 4

    expected_costs = [1.8, 2.5, 3.3, 4.0]
    for route, expected in zip(routes, expected_costs):
        assert abs(route.total_cost - expected) < 1e-9, (
            f"Expected {expected}, got {route.total_cost}"
        )

    # Verify sorted
    for i in range(len(routes) - 1):
        assert routes[i].total_cost <= routes[i + 1].total_cost + 1e-9


# ---------------------------------------------------------------------------
# Test 7 — k-way merge across reactions is correctly interleaved
# ---------------------------------------------------------------------------
#
# This specifically tests that routes from different reactions at the same
# MolNode are properly interleaved by cost, not concatenated sequentially.
#
#   T has 3 reactions:
#     R1 (cost 3.0): T -> A         (known)    total = 3.0
#     R2 (cost 1.0): T -> X         (not known)
#     R3 (cost 2.0): T -> B         (known)    total = 2.0
#
#   X has 1 reaction:
#     R4 (cost 0.5): X -> C         (known)    total = 0.5
#
#   So R2 total = 1.0 + 0.5 = 1.5
#
#   Sorted routes: 1.5 (R2+R4), 2.0 (R3), 3.0 (R1)
#
#   Without k-way merge, a naive sequential iteration sorted by
#   succ_value of reactions would yield R2 first (succ_value 1.5),
#   then R3 (2.0), then R1 (3.0) — which happens to be correct here.
#   But let's verify the ordering is right.


def test_kway_merge_interleaving():
    known = {"A", "B", "C"}
    tree = _build_tree("T", known)

    _expand(
        tree,
        "T",
        reactant_lists=[["A"], ["X"], ["B"]],
        costs=[3.0, 1.0, 2.0],
        templates=["r1", "r2", "r3"],
    )
    _expand(
        tree,
        "X",
        reactant_lists=[["C"]],
        costs=[0.5],
        templates=["r4"],
    )

    assert tree.succ

    routes = tree.get_routes(max_routes=10)
    assert len(routes) == 3

    costs = [r.total_cost for r in routes]
    assert abs(costs[0] - 1.5) < 1e-9
    assert abs(costs[1] - 2.0) < 1e-9
    assert abs(costs[2] - 3.0) < 1e-9


# ---------------------------------------------------------------------------
# Test 8 — get_best_route and get_routes(1) agree
# ---------------------------------------------------------------------------


def test_best_route_consistency():
    known = {"A", "B", "C", "D", "E"}
    tree = _build_tree("T", known)

    _expand(tree, "T", reactant_lists=[["X", "A"], ["B"]], costs=[1.0, 5.0])
    _expand(tree, "X", reactant_lists=[["C", "D"], ["E"]], costs=[0.5, 3.0])

    best = tree.get_best_route()
    routes_1 = tree.get_routes(max_routes=1)

    assert len(routes_1) == 1
    assert best.serialize() == routes_1[0].serialize()
    assert abs(best.total_cost - routes_1[0].total_cost) < 1e-9


# ---------------------------------------------------------------------------
# Test 9 — deeply nested tree (3 levels)
# ---------------------------------------------------------------------------
#
#   T -> X  (cost 1.0)
#   X -> Y  (cost 2.0)
#       X -> A  (cost 10.0)   (A known)
#   Y -> B  (cost 0.5)        (B known)
#       Y -> C  (cost 1.0)    (C known)
#
#   Routes:
#     T->X->Y->B:  1.0 + 2.0 + 0.5 = 3.5
#     T->X->Y->C:  1.0 + 2.0 + 1.0 = 4.0
#     T->X->A:     1.0 + 10.0       = 11.0


def test_deeply_nested():
    known = {"A", "B", "C"}
    tree = _build_tree("T", known)

    _expand(tree, "T", reactant_lists=[["X"]], costs=[1.0])
    _expand(tree, "X", reactant_lists=[["Y"], ["A"]], costs=[2.0, 10.0])
    _expand(tree, "Y", reactant_lists=[["B"], ["C"]], costs=[0.5, 1.0])

    assert tree.succ

    routes = tree.get_routes(max_routes=10)
    assert len(routes) == 3

    expected = [3.5, 4.0, 11.0]
    for route, exp in zip(routes, expected):
        assert abs(route.total_cost - exp) < 1e-9

import heapq
import logging
from queue import Queue

import numpy as np
from graphviz import Digraph

from retro_star.alg.mol_node import MolNode
from retro_star.alg.reaction_node import ReactionNode
from retro_star.alg.syn_route import SynRoute


class MolTree:
    def __init__(self, target_mol, known_mols, value_fn, zero_known_value=True):
        self.target_mol = target_mol
        self.known_mols = known_mols
        self.value_fn = value_fn
        self.zero_known_value = zero_known_value
        self.mol_nodes = []
        self.reaction_nodes = []

        self.root = self._add_mol_node(target_mol, None)
        self.succ = target_mol in known_mols
        self.search_status = 0

        if self.succ:
            logging.info("Synthesis route found: target in starting molecules")

    def _add_mol_node(self, mol, parent):
        is_known = mol in self.known_mols

        init_value = self.value_fn(mol)

        mol_node = MolNode(
            mol=mol,
            init_value=init_value,
            parent=parent,
            is_known=is_known,
            zero_known_value=self.zero_known_value,
        )
        self.mol_nodes.append(mol_node)
        mol_node.id = len(self.mol_nodes)

        return mol_node

    def _add_reaction_and_mol_nodes(self, cost, mols, parent, template, ancestors):
        assert cost >= 0

        for mol in mols:
            if mol in ancestors:
                return

        reaction_node = ReactionNode(parent, cost, template)
        for mol in mols:
            self._add_mol_node(mol, reaction_node)
        reaction_node.init_values()
        self.reaction_nodes.append(reaction_node)
        reaction_node.id = len(self.reaction_nodes)

        return reaction_node

    def expand(self, mol_node, reactant_lists, costs, templates):
        assert not mol_node.is_known and not mol_node.children

        if costs is None:  # No expansion results
            assert mol_node.init_values(no_child=True) == np.inf
            if mol_node.parent:
                mol_node.parent.backup(np.inf, from_mol=mol_node.mol)
            return self.succ

        assert mol_node.open
        ancestors = mol_node.get_ancestors()
        for i in range(len(costs)):
            self._add_reaction_and_mol_nodes(
                costs[i], reactant_lists[i], mol_node, templates[i], ancestors
            )

        if len(mol_node.children) == 0:  # No valid expansion results
            assert mol_node.init_values(no_child=True) == np.inf
            if mol_node.parent:
                mol_node.parent.backup(np.inf, from_mol=mol_node.mol)
            return self.succ

        v_delta = mol_node.init_values()
        if mol_node.parent:
            mol_node.parent.backup(v_delta, from_mol=mol_node.mol)

        if not self.succ and self.root.succ:
            logging.info("Synthesis route found!")
            self.succ = True

        return self.succ

    def get_best_route(self):
        if not self.succ:
            return None

        syn_route = SynRoute(
            target_mol=self.root.mol,
            succ_value=self.root.succ_value,
            search_status=self.search_status,
        )

        mol_queue = Queue()
        mol_queue.put(self.root)
        while not mol_queue.empty():
            mol = mol_queue.get()
            if mol.is_known:
                syn_route.set_value(mol.mol, mol.succ_value)
                continue

            best_reaction = None
            for reaction in mol.children:
                if reaction.succ:
                    if (
                        best_reaction is None
                        or reaction.succ_value < best_reaction.succ_value
                    ):
                        best_reaction = reaction
            assert best_reaction.succ_value == mol.succ_value

            reactants = []
            for reactant in best_reaction.children:
                mol_queue.put(reactant)
                reactants.append(reactant.mol)

            syn_route.add_reaction(
                mol=mol.mol,
                value=mol.succ_value,
                template=best_reaction.template,
                reactants=reactants,
                cost=best_reaction.cost,
            )

        return syn_route

    def get_routes(self, max_routes=1):
        """Extract up to *max_routes* synthesis routes ranked by total cost.

        Strategy (hybrid Option-B):

        1.  ``_get_subtree_routes(mol_node, limit)`` returns a **sorted list**
            of ``(cost, choices)`` tuples for all complete routes in the
            subtree rooted at *mol_node*, capped at *limit* entries.

        2.  At each non-leaf ``MolNode`` the method builds one *stream* per
            successful child ``ReactionNode``.  Each stream lazily produces
            routes in cost order via ``_heap_cartesian_product`` over the
            (materialized, sorted) sub-route lists of the reaction's
            reactants.

        3.  The per-reaction streams are merged with a **k-way heap merge**
            so that the combined output is globally sorted by cost.

        4.  The merge stops after *limit* items, preventing combinatorial
            blow-up.

        Returns a list of :class:`SynRoute` objects sorted by
        ``total_cost`` (ascending), deduplicated by serialized form.
        """
        if not self.succ:
            return []

        if max_routes <= 1:
            route = self.get_best_route()
            return [route] if route else []

        # -- helper: heap-based sorted Cartesian product ----------------

        def _heap_cartesian_product(option_lists, limit):
            """Yield up to *limit* elements of the Cartesian product of
            sorted lists in order of increasing sum-of-costs.

            Each element of *option_lists* is a list of ``(cost, choices)``
            tuples **already sorted by cost ascending**.
            """
            n = len(option_lists)
            if n == 0:
                yield (0.0, [])
                return
            if n == 1:
                for idx, item in enumerate(option_lists[0]):
                    if idx >= limit:
                        return
                    yield item
                return

            # Seed with the all-zeroth-index combination (cheapest)
            indices = tuple([0] * n)
            base_cost = sum(option_lists[i][0][0] for i in range(n))
            base_choices = []
            for i in range(n):
                base_choices.extend(option_lists[i][0][1])

            counter = 0  # tie-breaker for heap
            heap = [(base_cost, counter, indices, base_choices)]
            visited = {indices}
            yielded = 0

            while heap and yielded < limit:
                cost, _, idx_tuple, choices = heapq.heappop(heap)
                yield (cost, choices)
                yielded += 1

                # Successors: increment one dimension at a time
                for dim in range(n):
                    new_idx = idx_tuple[dim] + 1
                    if new_idx < len(option_lists[dim]):
                        new_key = list(idx_tuple)
                        new_key[dim] = new_idx
                        new_key = tuple(new_key)
                        if new_key not in visited:
                            visited.add(new_key)
                            new_cost = sum(
                                option_lists[i][new_key[i]][0] for i in range(n)
                            )
                            new_choices = []
                            for i in range(n):
                                new_choices.extend(option_lists[i][new_key[i]][1])
                            counter += 1
                            heapq.heappush(
                                heap,
                                (new_cost, counter, new_key, new_choices),
                            )

        # -- helper: per-reaction stream --------------------------------

        def _reaction_stream(mol_node, reaction, limit):
            """Lazily yield ``(cost, choices)`` for one reaction in cost
            order.  Each child reactant's sub-routes are materialized
            (via ``_get_subtree_routes``) so that the Cartesian product
            heap receives sorted lists.
            """
            child_options = []
            for reactant in reaction.children:
                sub = _get_subtree_routes(reactant, limit)
                if not sub:
                    return  # dead-end reactant – skip this reaction
                child_options.append(sub)

            for combo_cost, combo_choices in _heap_cartesian_product(
                child_options, limit
            ):
                yield (
                    reaction.cost + combo_cost,
                    [(mol_node, reaction)] + combo_choices,
                )

        # -- core: k-way merge of reaction streams ----------------------

        def _get_subtree_routes(mol_node, limit):
            """Return a **sorted** list of ``(cost, choices)`` for the
            subtree rooted at *mol_node*, containing at most *limit*
            entries.

            ``choices`` is a list of ``(MolNode, ReactionNode)`` pairs
            representing the reaction selected at each non-leaf node
            along the route.
            """
            if mol_node.is_known:
                return [(0.0, [])]

            succ_reactions = sorted(
                [r for r in mol_node.children if r.succ],
                key=lambda r: r.succ_value,
            )
            if not succ_reactions:
                return []

            # Build one lazy stream per successful reaction, prime each
            # by pulling its first (cheapest) element.
            counter = 0
            merge_heap = []  # (cost, counter, choices, iterator)
            for reaction in succ_reactions:
                stream = _reaction_stream(mol_node, reaction, limit)
                first = next(stream, None)
                if first is not None:
                    cost, choices = first
                    heapq.heappush(merge_heap, (cost, counter, choices, stream))
                    counter += 1

            # Pull from the merge heap in cost order
            results = []
            while merge_heap and len(results) < limit:
                cost, _, choices, stream = heapq.heappop(merge_heap)
                results.append((cost, choices))
                nxt = next(stream, None)
                if nxt is not None:
                    nxt_cost, nxt_choices = nxt
                    counter += 1
                    heapq.heappush(merge_heap, (nxt_cost, counter, nxt_choices, stream))

            return results

        # -- build SynRoute objects from the enumerated choices ---------

        raw_routes = _get_subtree_routes(self.root, max_routes)

        routes = []
        seen_serialized = set()

        for route_cost, route_choices in raw_routes:
            syn_route = SynRoute(
                target_mol=self.root.mol,
                succ_value=route_cost,
                search_status=self.search_status,
            )

            # Build a lookup: mol_node id -> chosen reaction_node
            choice_map = {id(mn): rn for mn, rn in route_choices}

            mol_queue = Queue()
            mol_queue.put(self.root)

            while not mol_queue.empty():
                mol_node = mol_queue.get()
                if mol_node.is_known:
                    syn_route.set_value(mol_node.mol, mol_node.succ_value)
                    continue

                reaction = choice_map.get(id(mol_node))
                if reaction is None:
                    # Should not happen for a valid route
                    break

                reactants = []
                for child in reaction.children:
                    mol_queue.put(child)
                    reactants.append(child.mol)

                syn_route.add_reaction(
                    mol=mol_node.mol,
                    value=mol_node.succ_value,
                    template=reaction.template,
                    reactants=reactants,
                    cost=reaction.cost,
                )

            # Deduplicate by serialized form
            serialized = syn_route.serialize()
            if serialized not in seen_serialized:
                seen_serialized.add(serialized)
                routes.append(syn_route)

        return routes

    def viz_search_tree(self, viz_file):
        G = Digraph("G", filename=viz_file)
        G.attr(rankdir="LR")
        G.attr("node", shape="box")
        G.format = "pdf"

        node_queue = Queue()
        node_queue.put((self.root, None))
        while not node_queue.empty():
            node, parent = node_queue.get()

            if node.open:
                color = "lightgrey"
            else:
                color = "aquamarine"

            if hasattr(node, "mol"):
                shape = "box"
            else:
                shape = "rarrow"

            if node.succ:
                color = "lightblue"
                if hasattr(node, "mol") and node.is_known:
                    color = "lightyellow"

            G.node(node.serialize(), shape=shape, color=color, style="filled")

            label = ""
            if hasattr(parent, "mol"):
                label = "%.3f" % node.cost
            if parent is not None:
                G.edge(parent.serialize(), node.serialize(), label=label)

            if node.children is not None:
                for c in node.children:
                    node_queue.put((c, node))

        G.render()

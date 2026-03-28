# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Graph placement helpers for the AIE backend."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from hls4ml.model.optimizer.optimizer import ModelOptimizerPass

from ..ir import get_backend_context
from ..op_impls import OpImplFootprint, OpImplPlacementContext

log = logging.getLogger(__name__)


@dataclass
class Rect:
    """Rectangular footprint of a node on the AIE grid."""

    w: int
    h: int
    in_col_off: int
    in_row_off: int
    out_col_off: int
    out_row_off: int


@dataclass
class NodeAdapter:
    """Thin adapter around an IR node for placement."""

    node: Any
    name: str
    rect: Rect
    anchor: Optional[Tuple[int, int]] = None  # local (x,y) if fixed


@dataclass
class Placed:
    """Concrete placement of a node in local grid coordinates."""

    name: str
    x: int
    y: int
    rect: Rect

    @property
    def in_abs(self) -> Tuple[int, int]:
        return (self.x + self.rect.in_col_off, self.y + self.rect.in_row_off)

    @property
    def out_abs(self) -> Tuple[int, int]:
        return (self.x + self.rect.out_col_off, self.y + self.rect.out_row_off)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _rects_conflict(candidate: Placed, existing: Placed) -> bool:
    """
    Check whether 'candidate' (with a one-column left margin) conflicts with 'existing'.

    Margin: [x-1, x+w-1] × [y, y+h-1], clipped at column 0.
    Existing: normal [x, x+w-1] × [y, y+h-1].
    Constraint for dense layer placement to avoid memory bank conflicts.
    """
    ax0 = max(0, candidate.x - 1)
    ax1 = candidate.x + candidate.rect.w - 1
    ay0 = candidate.y
    ay1 = candidate.y + candidate.rect.h - 1

    bx0 = existing.x
    bx1 = existing.x + existing.rect.w - 1
    by0 = existing.y
    by1 = existing.y + existing.rect.h - 1

    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def _in_bounds(p: Placed, W: int, H: int) -> bool:
    if p.x < 0 or p.y < 0:
        return False
    if p.x + p.rect.w > W:
        return False
    if p.y + p.rect.h > H:
        return False

    r = p.rect
    # pin offsets must lie inside rect
    return 0 <= r.in_col_off < r.w and 0 <= r.out_col_off < r.w and 0 <= r.in_row_off < r.h and 0 <= r.out_row_off < r.h


def _feasible(p: Placed, partial: List[Placed], W: int, H: int) -> bool:
    if not _in_bounds(p, W, H):
        return False
    return all(not _rects_conflict(p, q) for q in partial)


# ---------------------------------------------------------------------------
# Cost model (chain)
# ---------------------------------------------------------------------------


def _placement_cost_chain(placed: List[Placed], lam: float, mu: float) -> float:
    """
    Total cost for the placed chain so far.

    - horizontal: sum |c_out - c_in|
    - vertical:   sum |r_out - r_in|
    - bias:       mu * sum(top_row_y)

    Note: local coordinates are used (0..H-1).
    """
    horiz = 0.0
    vert = 0.0
    sum_rows = 0.0

    for i in range(len(placed) - 1):
        c_out, r_out = placed[i].out_abs
        c_in, r_in = placed[i + 1].in_abs
        horiz += abs(c_out - c_in)
        # TODO for direct connections we must use both vertical costs
        # vert  += abs(r_out - r_in)
        vert += abs(r_in - 0)

    for p in placed:
        sum_rows += p.y

    return horiz + lam * vert + mu * sum_rows


# ---------------------------------------------------------------------------
# Branch-and-bound placement
# ---------------------------------------------------------------------------


def _bnb_place_chain(
    chain: List[NodeAdapter],
    W: int,
    H: int,
    lam: float,
    mu: float,
    col_window: int = 6,
) -> List[Placed]:
    """Place a linear chain of rectangles using branch-and-bound."""

    if not chain:
        return []

    # First node: if it has an anchor, honor it; otherwise start at (0,0)
    first_spec = chain[0]
    start_x, start_y = first_spec.anchor if first_spec.anchor is not None else (0, 0)
    first = Placed(first_spec.name, start_x, start_y, first_spec.rect)
    if not _in_bounds(first, W, H):
        raise ValueError(f'Placement anchor for {first_spec.name} is out of bounds.')

    best: List[Placed] = []
    best_cost: float = math.inf

    def optimistic_bound(partial: List[Placed]) -> float:
        # Exact cost for the prefix, assuming zero cost for remaining nodes (admissible).
        return _placement_cost_chain(partial, lam, mu)

    def candidate_positions(prev: Placed, rect: Rect) -> Iterable[Tuple[int, int]]:
        """
        Enumerate candidate (x,y) positions for 'rect', ordered by proximity in x
        to the previous rect's end (ideal_x). We restrict x to a window around
        ideal_x to keep branching under control.
        """
        ideal_x = prev.x + prev.rect.w  # just to the right of previous box

        min_x = max(0, ideal_x - col_window)
        max_x = min(W - rect.w, ideal_x + col_window)
        if min_x > max_x:
            # degenerate / tiny grid: fall back to full search
            min_x, max_x = 0, W - rect.w

        xs = list(range(min_x, max_x + 1))
        xs.sort(key=lambda c: abs(c - ideal_x))

        ys = range(0, H - rect.h + 1)
        for y in ys:
            for x in xs:
                yield x, y

    def dfs(partial: List[Placed], idx: int) -> None:
        nonlocal best, best_cost

        if idx == len(chain):
            tot = _placement_cost_chain(partial, lam, mu)
            if tot < best_cost:
                best_cost = tot
                best = partial.copy()
            return

        if optimistic_bound(partial) >= best_cost:
            return

        prev = partial[-1]
        spec = chain[idx]

        # If this node has a fixed anchor, try only that.
        if spec.anchor is not None:
            ax, ay = spec.anchor
            cand = Placed(spec.name, ax, ay, spec.rect)
            if _feasible(cand, partial, W, H):
                partial.append(cand)
                dfs(partial, idx + 1)
                partial.pop()
            return

        for x, y in candidate_positions(prev, spec.rect):
            cand = Placed(spec.name, x, y, spec.rect)
            if not _feasible(cand, partial, W, H):
                continue
            partial.append(cand)
            dfs(partial, idx + 1)
            partial.pop()

    # Optional: seed with a greedy solution to tighten pruning
    # best: List[Placed] = []
    # best_cost: float = math.inf
    seed = _greedy_right_first(chain, W, H)
    if seed:
        best = seed
        best_cost = _placement_cost_chain(seed, lam, mu)

    dfs([first], 1)

    if not best:
        raise RuntimeError('No feasible placement found for the given chain and grid.')

    return best


def _greedy_right_first(chain: List[NodeAdapter], W: int, H: int) -> List[Placed]:
    """Simple greedy heuristic: place each rect to the right, then first-fit."""
    if not chain:
        return []

    g0 = chain[0]
    x0, y0 = g0.anchor if g0.anchor is not None else (0, 0)
    placed: List[Placed] = []
    cur = Placed(g0.name, x0, y0, g0.rect)
    if not _in_bounds(cur, W, H):
        return []
    placed.append(cur)

    for spec in chain[1:]:
        # prefer right of previous
        prev = placed[-1]
        try_x = prev.x + prev.rect.w
        try_y = prev.y
        cand = Placed(spec.name, try_x, try_y, spec.rect)
        if _feasible(cand, placed, W, H):
            placed.append(cand)
            continue

        # fallback scan
        found = False
        for y in range(0, H - spec.rect.h + 1):
            for x in range(0, W - spec.rect.w + 1):
                cand = Placed(spec.name, x, y, spec.rect)
                if _feasible(cand, placed, W, H):
                    placed.append(cand)
                    found = True
                    break
            if found:
                break

        if not found:
            return []

    return placed


class PlaceKernels(ModelOptimizerPass):
    """
    Assign tile coordinates to rectangular compute layers
    using a branch-and-bound placement algorithm.

    - Uses config.parameters.parallelism.cas_length as width and
      config.parameters.parallelism.cas_num as height.
    - Honors user placement hints from node.directives['placement'] (col, row):
      for the first dense layer this becomes the anchor; later layers
      can also be forced to a fixed tile.
    """

    def __init__(self, lam: float = 1.0, mu: float = 0.05, col_window: int = 6):
        self.name = 'place_kernels'
        self._lam = float(lam)
        self._mu = float(mu)
        self._col_window = int(col_window)

    def transform(self, model) -> bool:
        ctx = get_backend_context(model)
        device = ctx.device

        # local coordinates: [0 .. W-1], [0 .. H-1]
        W = int(device.columns)
        H = int(device.rows)
        col_offset = int(device.column_start)
        row_offset = int(device.row_start)

        kernel_nodes = [(node, ctx.ir.execution.get(node.name)) for node in ctx.ir.logical]
        kernel_nodes = [(node, inst) for node, inst in kernel_nodes if inst]
        if not kernel_nodes:
            return False

        chain: List[NodeAdapter] = []
        for node, inst in kernel_nodes:
            footprint = self._node_footprint(ctx, node, inst)
            w = footprint.width
            h = footprint.height

            # For now, approximate I/O ports as middle-left and middle-right.
            in_row = max(0, min(h - 1, h // 2))
            rect = Rect(
                w=w,
                h=h,
                in_col_off=0,
                in_row_off=in_row,
                out_col_off=w - 1,
                out_row_off=in_row,
            )

            # Optional user placement hint (absolute coords)
            anchor = None
            placement_hint = node.directives.get('placement', {})
            if placement_hint:
                hint_col = placement_hint.get('col')
                hint_row = placement_hint.get('row')
                if hint_col is not None and hint_row is not None:
                    # convert absolute → local
                    ax = int(hint_col) - col_offset
                    ay = int(hint_row) - row_offset
                    anchor = (ax, ay)

            chain.append(NodeAdapter(node=node, name=node.name, rect=rect, anchor=anchor))

        try:
            placed_chain = _bnb_place_chain(
                chain,
                W=W,
                H=H,
                lam=self._lam,
                mu=self._mu,
                col_window=self._col_window,
            )
        except Exception as exc:
            log.warning('AIE placement: branch-and-bound failed (%s); falling back to legacy greedy layout.', exc)
            return self._legacy_fallback(ctx, col_offset, row_offset, W, H)

        # Map placements back to nodes by name
        placed_by_name: Dict[str, Placed] = {p.name: p for p in placed_chain}

        changed = False

        for nad in chain:
            node = nad.node
            placed = placed_by_name[node.name]
            col = placed.x + col_offset
            row = placed.y + row_offset
            placement = {'col': int(col), 'row': int(row)}

            prev = ctx.ir.physical.placements.get(node.name)
            if prev != placement:
                ctx.ir.physical.placements[node.name] = placement.copy()
                changed = True

        return changed

    # Optional: legacy greedy fallback
    @staticmethod
    def _legacy_fallback(ctx, col_offset: int, row_offset: int, W: int, H: int) -> bool:
        """Very simple greedy placement pass."""
        current_col = col_offset
        current_row = row_offset
        max_col = col_offset + W
        max_row = row_offset + H

        changed = False

        for node in ctx.ir.logical:
            inst = ctx.ir.execution.get(node.name)
            if inst is not None:
                footprint = PlaceKernels._footprint_static(ctx, node, inst)
                width = footprint.width
                height = footprint.height
                requested = node.directives.get('placement', {})
                requested_col = requested.get('col')
                requested_row = requested.get('row')
                user_specified = requested_col is not None or requested_row is not None

                if user_specified:
                    col = int(requested_col) if requested_col is not None else current_col
                    row = int(requested_row) if requested_row is not None else current_row
                    placement = {'col': col, 'row': row}
                    current_col, current_row = col + width + 1, row
                else:
                    if current_col + width + 1 > max_col:
                        current_col = col_offset
                        current_row += height
                    if current_row + height > max_row:
                        log.warning(
                            'AIE backend: node %s exceeds grid dimensions (%dx%d starting at %d,%d).',
                            node.name,
                            W,
                            H,
                            col_offset,
                            row_offset,
                        )
                    placement = {'col': current_col, 'row': current_row}
                    current_col += width + 1

                prev = ctx.ir.physical.placements.get(node.name)
                if prev != placement:
                    ctx.ir.physical.placements[node.name] = placement.copy()
                    changed = True

        return changed

    @staticmethod
    def _footprint_static(ctx, node, inst=None) -> 'OpImplFootprint':
        if inst is None:
            inst = ctx.ir.execution.get(node.name)
        if inst is None:
            raise RuntimeError(f'{node.name}: op implementation missing; run resolve before placement.')
        config = inst.config
        placement_ctx = OpImplPlacementContext(
            node=node,
            metadata=node.metadata,
            config=config,
        )
        footprint = inst.variant.footprint(placement_ctx)
        if footprint is None:
            raise RuntimeError(f'{node.name}: implementation variant did not provide a footprint.')
        return footprint

    def _node_footprint(self, ctx, node, inst):
        return self._footprint_static(ctx, node, inst)

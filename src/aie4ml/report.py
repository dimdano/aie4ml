# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Collect an AIE project's resource, latency and per-kernel cycle data into one report.

The numbers come from artifacts the AMD toolchain leaves in the project, which appear in
tiers -- the report gathers whatever is present and says what is missing:

    make aiecom     Work/reports/*          tiles, program/stack/heap memory
    make aiesim     aiesimulator_output/    output interval, PLIO throughput
    make profile    profile_funct_*.txt     per-kernel cycles

`make aiesim` alone does not profile, so a plain `predict(simulator='aie')` run yields
latency and throughput but no per-kernel breakdown. Re-run `make profile` for that.

    from aie4ml.report import report        # notebook: report(model) or report(project_dir)
    aie4ml-report path/to/project-folder    # terminal
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Itanium mangling prefixes each identifier with its length (_ZN12dense_singleI... ->
# 12 chars of 'dense_single'), which is how the class name is recovered from the symbol.
_KERNEL_RE = re.compile(r'run _ZN(\d+)([A-Za-z_][A-Za-z0-9_]*)')
_PLIO_RE = re.compile(r'\|\s*(?:plio)?\s*\|?\s*(PLIO_\w+)\s*\|\s*(IN|OUT)\s*\|\s*([\d.]+)')
_CORE_RE = re.compile(r'^Core (\S+)', re.M)

ASSUMED_AIE_CLOCK_GHZ = 1.25


def _analyze_aie_out_interval(output_dir: Path) -> Dict:
    data_dir = output_dir / 'aiesimulator_output' / 'data'

    if not data_dir.exists():
        return {}

    per_file = {}
    all_lat = []
    first_out = []

    for fp in sorted(data_dir.glob('y_p*.txt')):
        stamps = _timestamps(fp)
        if stamps:
            first_out.append(min(stamps))
        lst = _parse_timing(fp)
        if lst:
            per_file[fp.name] = {
                'min_ns': round(min(lst), 3),
                'max_ns': round(max(lst), 3),
                'avg_ns': round(sum(lst) / len(lst), 3),
                'samples': len(lst),
            }
            all_lat.extend(lst)

    if not all_lat:
        return {}

    return {
        'global': {
            'min_ns': round(min(all_lat), 3),
            'max_ns': round(max(all_lat), 3),
            'avg_ns': round(sum(all_lat) / len(all_lat), 3),
            'samples': len(all_lat),
        },
        # When the first result leaves the graph: one sample's whole trip, including the DMA
        # and memtile hops that the per-kernel cycle counts do not see.
        'first_output_ns': round(min(first_out), 3) if first_out else None,
        'per_port': per_file,
    }


def _timestamps(path: Path) -> List[float]:
    """Absolute 'T <value> <unit>' marks the simulator writes ahead of each output line."""
    unit_ns = {'ps': 1e-3, 'ns': 1.0, 'us': 1e3, 'ms': 1e6}
    out = []
    with open(path) as handle:
        for line in handle:
            m = re.match(r'T\s+([\d.]+)\s*(ps|ns|us|ms)', line)
            if m:
                out.append(float(m.group(1)) * unit_ns[m.group(2)])
    return out


def _parse_timing(path: Path) -> List[float]:
    """Return TLAST-to-TLAST intervals (in nanoseconds)."""
    regex = re.compile(r'^T\s+(\d+)\s*(ps|ns|us|ms|s)', re.IGNORECASE)

    lat = []
    last_tlast_time = None
    current_time = None

    with open(path) as f:
        for line in f:
            line = line.strip()

            m = regex.match(line)
            if m:
                val, unit = m.groups()
                current_time = _convert_to_ns(int(val), unit)
                continue

            if 'TLAST' in line.upper():
                if last_tlast_time is not None and current_time is not None:
                    dt = current_time - last_tlast_time
                    if dt >= 0:
                        lat.append(dt)
                last_tlast_time = current_time

    return lat


def _convert_to_ns(value: int, unit: str) -> float:
    if unit == 'ps':
        return value / 1000
    if unit == 'ns':
        return value
    if unit == 'us':
        return value * 1000
    if unit == 'ms':
        return value * 1_000_000
    if unit == 's':
        return value * 1_000_000_000
    raise ValueError(f'Unknown time unit: {unit}')


def _plan_ops(doc: Dict[str, Any]) -> int:
    """MAC-equivalent ops per inference, from the plan rather than a live model.

    Only dense/matmul contribute; each output element costs one multiply and one add over
    the reduction, so 2 * n_in * n_out * (elements outside the feature axis).
    """
    shapes = {}
    for entry in doc.get('execution', []):
        for name, view in ((entry.get('config') or {}).get('io_views') or {}).items():
            shapes[name] = view.get('logical') or []
    ops = 0
    for node in doc.get('logical', []):
        if node.get('op_type') not in ('dense', 'matmul'):
            continue
        meta = node.get('metadata') or {}
        out = shapes.get((node.get('outputs') or [None])[0]) or []
        if not out or 'n_in' not in meta:
            continue
        independent = 1
        for dim in out[:-1]:
            independent *= int(dim)
        ops += 2 * int(meta['n_in']) * int(meta['n_out']) * independent
    return ops


def _critical_path(edges: List[tuple], stage_cycles: Dict[str, int]) -> Dict[str, Any]:
    """Longest dependency chain, weighted by each stage's measured cycles."""
    preds: Dict[str, List[str]] = {op: [] for op in stage_cycles}
    for src, dst in edges:
        preds.setdefault(dst, []).append(src)
        preds.setdefault(src, [])

    order, seen = [], set()

    def visit(node, guard=()):  # depth-first, tolerant of an unexpected cycle
        if node in seen or node in guard:
            return
        for p in preds.get(node, []):
            visit(p, guard + (node,))
        seen.add(node)
        order.append(node)

    for op in preds:
        visit(op)

    best: Dict[str, int] = {}
    came: Dict[str, Optional[str]] = {}
    for name in order:
        prior = max(((best.get(p, 0), p) for p in preds.get(name, [])), default=(0, None))
        best[name] = prior[0] + stage_cycles.get(name, 0)
        came[name] = prior[1]
    if not best:
        return {}
    end = max(best, key=lambda n: best[n])
    chain, node = [], end
    while node is not None:
        chain.append(node)
        node = came.get(node)
    return {'cycles': best[end], 'chain': list(reversed(chain))}


def _vitis(project: Path) -> Dict[str, Any]:
    """Design facts straight from the AIE compiler's own report."""
    path = project / 'Work' / 'reports' / 'compiler_report.json'
    if not path.exists():
        return {}
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}

    blocks = doc.get('blockInstances', {})
    placed = doc.get('mapping', {}).get('blockInstanceMapping', {})
    tiles: Dict[str, Dict[str, Any]] = {}
    memtiles: List[Dict[str, Any]] = []
    inst_op: Dict[str, str] = {}

    for inst, where in placed.items():
        meta = blocks.get(inst, {})
        core = where.get('coreInfo')
        if core and core.get('tile') == 'aie':
            # qualifiedGraphName is dut.dut.<op>; the last component names the op.
            op = (meta.get('qualifiedGraphName') or inst).split('.')[-1]
            inst_op[inst] = op
            tiles[f'{core["column"]}_{core["row"]}'] = {'op': op, 'graph': meta.get('graphName')}
            continue
        for buf in where.get('bufferInfo') or []:
            if buf.get('tile') == 'memory':
                memtiles.append(
                    {
                        'name': (meta.get('qualifiedName') or inst).split('.')[-1],
                        'tile': f'{buf["column"]}_{buf["row"]}',
                        'bytes': int(buf.get('size', 0)),
                    }
                )
                break

    nets = [
        (n['srcInstance'], n['dstInstance'])
        for n in doc.get('nets', {}).values()
        if 'srcInstance' in n and 'dstInstance' in n
    ]
    return {'tiles': tiles, 'memtiles': memtiles, 'nets': nets, 'inst_op': inst_op}


def _op_edges(vitis: Dict[str, Any]) -> List[tuple]:
    """Op-to-op dependencies, hopping over the buffers and DMAs that sit between them."""
    inst_op = vitis.get('inst_op') or {}
    succ: Dict[str, List[str]] = {}
    for src, dst in vitis.get('nets') or []:
        succ.setdefault(src, []).append(dst)

    edges = set()
    for src in inst_op:
        seen, stack = set(), list(succ.get(src, []))
        while stack:  # walk forward until the next core instance
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            if node in inst_op:
                edges.add((inst_op[src], inst_op[node]))
            else:
                stack.extend(succ.get(node, []))
    return sorted(edges)


def _project_dir(model_or_path) -> Path:
    if hasattr(model_or_path, '_aie_backend_context'):
        from .ir import get_backend_context

        return Path(get_backend_context(model_or_path).project_config.output_dir).resolve()
    return Path(model_or_path).resolve()


def _kernel_cycles(project: Path, vitis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per-tile compute cycles and stall share, from the --profile function reports."""
    owned = _op_tiles(project, vitis)
    rows = []
    for path in sorted(project.glob('aiesimulator_output/profile_funct_*.txt')):
        text = path.read_text(errors='ignore')
        run = next((ln for ln in text.splitlines() if 'run _ZN' in ln), None)
        if run is None:
            continue
        fields = run.split()
        try:
            calls, total = int(fields[0]), int(fields[1])
        except (IndexError, ValueError):
            continue
        match = _KERNEL_RE.search(run)
        kernel = match.group(2)[: int(match.group(1))] if match else '?'
        # Columns are: calls, cycles, %-of-report, min, avg, max. The kernel's own share of
        # the simulated window is its utilisation; the rest is time in main, i.e. blocked on
        # an input buffer. Both come from the same total, so only one is worth reporting.
        busy = fields[2].rstrip('%') if len(fields) > 2 else ''
        tile = path.stem.replace('profile_funct_', '')
        rows.append(
            {
                'tile': tile,
                **{k: v for k, v in owned.get(tile, {}).items()},
                'kernel': kernel,
                'calls': calls,
                'cycles_per_call': round(total / calls) if calls else 0,
                'busy_pct': float(busy) if busy else None,
            }
        )
    return rows


def _op_tiles(project: Path, vitis: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Map each tile "col_row" to the op that owns it."""
    owned: Dict[str, Dict[str, Any]] = {t: dict(v) for t, v in (vitis.get('tiles') or {}).items()}
    doc = _pipeline(project)
    if not doc:
        return owned
    placements = doc.get('physical', {}).get('placements', {})
    for entry in doc.get('execution', []):
        name = entry.get('node')
        at = placements.get(name)
        if not at:
            continue
        par = (entry.get('config') or {}).get('parallelism', {}) or {}
        num, length = int(par.get('cas_num', 1)), int(par.get('cas_length', 1))
        for dy in range(num):
            for dx in range(length):
                owned.setdefault(f'{at["col"] + dx}_{at["row"] + dy}', {}).update(
                    {
                        'op': name,
                        'variant': entry.get('variant_id', '?'),
                        'cas_num': num,
                        'cas_length': length,
                        'contract': par.get('contract', 'inner'),
                    }
                )
    return owned


def _bottleneck(kernels: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cycle budget for one iteration, one row per graph stage."""
    staged: Dict[str, Dict[str, Any]] = {}
    for k in kernels:
        op = k.get('op')
        if op is None:
            continue
        prev = staged.get(op)
        if prev is None or k['cycles_per_call'] > prev['cycles']:
            staged[op] = {'cycles': k['cycles_per_call'], 'kernel': k['kernel'], 'tiles': 0}
    if not staged:
        return {}
    for k in kernels:
        if k.get('op') in staged:
            staged[k['op']]['tiles'] += 1

    grand = sum(v['cycles'] for v in staged.values()) or 1
    for v in staged.values():
        v['share_pct'] = round(100.0 * v['cycles'] / grand, 1)
    ranked = dict(sorted(staged.items(), key=lambda kv: -kv[1]['cycles']))

    by_kernel: Dict[str, int] = {}
    for v in ranked.values():
        by_kernel[v['kernel']] = by_kernel.get(v['kernel'], 0) + v['cycles']
    return {
        'per_stage': ranked,
        'per_kernel_class': dict(sorted(by_kernel.items(), key=lambda kv: -kv[1])),
        'iteration_cycles': grand,
        'dominant': next(iter(ranked)),
    }


def _throughput(project: Path) -> List[Dict[str, Any]]:
    """PLIO port throughput, printed by the simulator at the end of the run."""
    log = project / 'log'
    if not log.exists():
        return []
    ports: Dict[str, Dict[str, Any]] = {}
    for line in log.read_text(errors='ignore').splitlines():
        m = _PLIO_RE.search(line)
        if m:
            ports[m.group(1)] = {'port': m.group(1), 'dir': m.group(2), 'MBps': float(m.group(3))}
    return list(ports.values())


def _memory_by_core(project: Path) -> Dict[str, Dict[str, int]]:
    """Per-core program/stack/heap usage, keyed by the same "col_row" the profiles use."""
    fields = (
        ('program', 'report_pm.txt', r'PM Size Used = (\d+)'),
        ('stack', 'report_stack.txt', r'Stack Size Used = (\d+)'),
        ('heap', 'report_heap.txt', r'Heap Size Used[^=]*= (\d+)'),
    )
    out: Dict[str, Dict[str, int]] = {}
    for kind, fname, pattern in fields:
        path = project / 'Work' / 'reports' / fname
        if not path.exists():
            continue
        for core, value in re.findall(r'Core (\S+)[\s\S]*?' + pattern, path.read_text(errors='ignore')):
            out.setdefault(core, {})[kind] = int(value)
    return out


def _memory(project: Path) -> Dict[str, Any]:
    """Program / stack / heap high-water marks across cores."""

    def scan(fname: str, used_re: str, alloc_re: str) -> Optional[Dict[str, Any]]:
        path = project / 'Work' / 'reports' / fname
        if not path.exists():
            return None
        text = path.read_text(errors='ignore')
        used = [int(x) for x in re.findall(used_re, text)]
        alloc = [int(x) for x in re.findall(alloc_re, text)]
        if not used:
            return None
        return {
            'cores': len(_CORE_RE.findall(text)),
            'max_used': max(used),
            'allotted': max(alloc) if alloc else None,
            'pct': round(100.0 * max(used) / max(alloc), 1) if alloc and max(alloc) else None,
        }

    return {
        k: v
        for k, v in {
            'program': scan('report_pm.txt', r'PM Size Used = (\d+)', r'PM Size Allotted = (\d+)'),
            'stack': scan('report_stack.txt', r'Stack Size Used = (\d+)', r'Stack Size Allotted = (\d+)'),
            'heap': scan('report_heap.txt', r'Heap Size Used[^=]*= (\d+)', r'Heap Size Allotted = (\d+)'),
        }.items()
        if v
    }


def _pipeline(project: Path) -> Dict[str, Any]:
    """The emitted pipeline description, or {} when the project was never written."""
    path = project / 'aie_pipeline.json'
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _design(project: Path, vitis: Dict[str, Any]) -> Dict[str, Any]:
    """Resource totals: AIE tiles occupied, memtile buffers staged through, distinct ops."""
    if vitis.get('tiles'):
        memtiles = vitis.get('memtiles') or []
        return {
            'aie_tiles': len(vitis['tiles']),
            'memtile_buffers': len(memtiles),
            'memtile_bytes': sum(m['bytes'] for m in memtiles),
            'ops': len({t['op'] for t in vitis['tiles'].values()}),
        }
    doc = _pipeline(project)
    if not doc:
        return {}

    def find(node, key):
        if isinstance(node, dict):
            if key in node:
                return node
            for value in node.values():
                hit = find(value, key)
                if hit is not None:
                    return hit
        if isinstance(node, list):
            for value in node:
                hit = find(value, key)
                if hit is not None:
                    return hit
        return None

    plan = find(doc, 'direct_edges') or {}
    aie_tiles = sum(
        int((e.get('config') or {}).get('parallelism', {}).get('cas_num', 1))
        * int((e.get('config') or {}).get('parallelism', {}).get('cas_length', 1))
        for e in doc.get('execution', [])
    )
    return {
        'aie_tiles': aie_tiles,
        'memtile_buffers': len(plan.get('buffers', [])),
        'ops': len(doc.get('execution', [])),
    }


class Report(dict):
    """The collected metrics.

    Prints as the rendered report and indexes as the raw dict, so a notebook cell showing
    `report(model)` displays the tables while `report(model)['bottleneck']` still works.
    """

    def __str__(self) -> str:
        return format_report(self)

    __repr__ = __str__


def report(model_or_path) -> 'Report':
    """Collect every available metric for a built AIE project.

    Accepts a model (hls4ml or aie4ml) or a path to the project directory.
    """
    return collect_report(model_or_path)


def collect_report(model_or_path) -> 'Report':
    """Gather every available metric for a built AIE project."""
    project = _project_dir(model_or_path)
    if not project.exists():
        raise FileNotFoundError(f'{project}: project directory not found.')

    vitis = _vitis(project)
    doc = _pipeline(project)
    kernels = _kernel_cycles(project, vitis)
    per_core = _memory_by_core(project)
    for k in kernels:
        k.update({f'{kind}_B': v for kind, v in per_core.get(k['tile'], {}).items()})
    latency = _analyze_aie_out_interval(project)
    report: Dict[str, Any] = {
        'project': str(project),
        'design': _design(project, vitis),
        'latency': latency,
        'throughput_plio': _throughput(project),
        'kernels': kernels,
        'bottleneck': _bottleneck(kernels),
        'memory': _memory(project),
    }

    interval = (latency or {}).get('global') or {}
    ops = _plan_ops(doc)
    if ops and interval.get('avg_ns'):
        # ns and GOP/s cancel: ops / ns == GOP/s.
        report['compute'] = {
            'ops_per_inference': ops,
            'avg_GOPs': round(ops / interval['avg_ns'], 2),
            'peak_GOPs': round(ops / interval['min_ns'], 2),
        }
    stage_cycles = {op: v['cycles'] for op, v in (report['bottleneck'].get('per_stage') or {}).items()}
    if stage_cycles:
        edges = _op_edges(vitis) or [
            (produced[t], node['name'])
            for node in doc.get('logical', [])
            for t in (node.get('inputs') or [])
            if (produced := {tn: n['name'] for n in doc.get('logical', []) for tn in (n.get('outputs') or [])})
            and t in produced
        ]
        critical = _critical_path(edges, stage_cycles)
        report['critical_path'] = critical
        first = (latency or {}).get('first_output_ns')
        if critical and first:
            # This is a residual, not a measurement: end-to-end minus the kernel work on the critical path.
            compute_ns = critical['cycles'] / ASSUMED_AIE_CLOCK_GHZ
            report['latency_split'] = {
                'assumed_clock_GHz': ASSUMED_AIE_CLOCK_GHZ,
                'total_ns': first,
                'compute_ns': round(compute_ns, 1),
                'data_movement_ns': round(first - compute_ns, 1),
                'data_movement_pct': round(100.0 * (first - compute_ns) / first, 1),
            }

    missing = []
    if not kernels:
        missing.append("per-kernel cycles: run 'make profile' (plain 'make aiesim' does not profile)")
    if not report['throughput_plio']:
        missing.append("PLIO throughput: run 'make aiesim' or 'make profile'")
    if not report['memory']:
        missing.append("memory: run 'make aiecom'")
    report['missing'] = missing
    return Report(report)


def format_report(report: Dict[str, Any]) -> str:
    """Render the report as plain text."""
    out: List[str] = []
    add = out.append
    ns = lambda cycles: cycles / ASSUMED_AIE_CLOCK_GHZ  # noqa: E731 - local unit shorthand

    add('=' * 78)
    add('  aie4ml  |  AIE project report')
    add(f'  {report["project"]}')
    add('=' * 78)

    design = report.get('design') or {}
    if design:
        line = (
            f'  {design["ops"]} ops on {design["aie_tiles"]} AIE tiles, ' f'{design["memtile_buffers"]} memtile buffers'
        )
        if design.get('memtile_bytes'):
            line += f' ({design["memtile_bytes"]:,} B)'
        add(line)
    add(f'  Clock freq. (assumed): {ASSUMED_AIE_CLOCK_GHZ} GHz')

    latency = (report.get('latency') or {}).get('global') or {}
    first = (report.get('latency') or {}).get('first_output_ns')
    critical = report.get('critical_path') or {}
    split = report.get('latency_split') or {}
    if first or critical:
        add('')
        add('Latency  (one sample, input to output)')
        if first:
            add(f'    end to end     {first * ASSUMED_AIE_CLOCK_GHZ:12,.0f} cc  {first:12,.1f} ns   measured')
        if critical:
            add(
                f'      compute      {critical["cycles"]:12,d} cc  {ns(critical["cycles"]):12,.1f} ns   '
                f'critical path, {len(critical["chain"])} stages'
            )
        if split:
            add(
                f'      data move    {split["data_movement_ns"] * ASSUMED_AIE_CLOCK_GHZ:12,.0f} cc  '
                f'{split["data_movement_ns"]:12,.1f} ns   {split["data_movement_pct"]}% memtile/DMA/lock etc.'
            )

    compute = report.get('compute') or {}
    if latency or compute:
        add('')
        add('Throughput  (steady state)')
        if latency:
            add(
                f'    output interval {latency["avg_ns"] * ASSUMED_AIE_CLOCK_GHZ:12,.0f} cc '
                f'{latency["avg_ns"]:12,.1f} ns   avg ({latency["min_ns"]:,.1f} min)'
            )
        if compute:
            add(
                f'    compute rate   {compute["avg_GOPs"]:12,.2f} GOP/s  '
                f'  {compute["ops_per_inference"]:,} ops/inference'
            )

    bottleneck = report.get('bottleneck') or {}
    if bottleneck:
        add('')
        add('Per-stage cost')
        add(f'    {"op":26s} {"kernel":22s} {"tiles":>5s} {"cc/iter":>9s} ' f'{"ns":>9s} {"busy":>6s} {"prog B":>7s}')
        rows = {k['op']: k for k in report.get('kernels') or []}
        for name, v in bottleneck['per_stage'].items():
            k = rows.get(name, {})
            busy = f'{k["busy_pct"]:5.1f}%' if k.get('busy_pct') is not None else '     -'
            add(
                f'    {name:26s} {v["kernel"]:22s} {v["tiles"]:5d} {v["cycles"]:9,d} '
                f'{ns(v["cycles"]):9,.1f} {busy} {k.get("program_B", 0):7,d}'
            )
        add('')
        add('    by kernel class')
        for cls, cyc in bottleneck['per_kernel_class'].items():
            add(
                f'        {cls:22s} {cyc:9,d} cc  {ns(cyc):9,.1f} ns  '
                f'{100.0 * cyc / bottleneck["iteration_cycles"]:5.1f}%'
            )

    plio = report.get('throughput_plio') or []
    if plio:
        add('')
        add('PLIO throughput')
        for p in plio:
            add(f'    {p["port"]:16s} {p["dir"]:4s} {p["MBps"]:10.2f} MBps')

    memory = report.get('memory') or {}
    if memory:
        add('')
        add('Memory high-water (worst core)')
        for kind, v in memory.items():
            pct = f'{v["pct"]:5.1f}%' if v.get('pct') is not None else '    -'
            add(f'    {kind:9s} {v["max_used"]:7,d} / {v["allotted"] or 0:7,d} B  {pct}  across {v["cores"]} cores')

    if report.get('missing'):
        add('')
        add('Not collected')
        for m in report['missing']:
            add(f'    - {m}')
    return '\n'.join(out)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('project', type=Path, help='AIE project directory (the output_dir used to build it)')
    parser.add_argument('--json', action='store_true', help='emit the raw report as JSON')
    args = parser.parse_args(argv)
    try:
        collected = report(args.project)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(json.dumps(collected, indent=2) if args.json else collected)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

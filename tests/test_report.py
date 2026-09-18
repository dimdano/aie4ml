from __future__ import annotations

import pytest
from aie4ml.report import _analyze_aie_out_interval


def _pipeline(elements: int) -> dict:
    return {
        'physical': {
            'plan': {
                'io_ports': [
                    {
                        'direction': 'output',
                        'port': 0,
                        'tensor': 'y',
                        'staging': {'io_tiling_dimension': [elements]},
                    }
                ]
            }
        }
    }


@pytest.mark.parametrize(
    'frames',
    [
        [(10, 8), (20, 8), (30, 8)],
        [(8, 4), (10, 4), (18, 4), (20, 4), (28, 4), (30, 4)],
    ],
    ids=['one-frame-per-inference', 'two-frames-per-inference'],
)
def test_report_groups_tlast_frames_into_logical_inferences(tmp_path, frames):
    data_dir = tmp_path / 'aiesimulator_output' / 'data'
    data_dir.mkdir(parents=True)
    lines = []
    for timestamp, elements in frames:
        lines.extend([f'T {timestamp} ns', 'TLAST', ' '.join(['1'] * elements)])
    (data_dir / 'y_p0.txt').write_text('\n'.join(lines) + '\n')

    latency = _analyze_aie_out_interval(tmp_path, _pipeline(8))

    assert latency['global'] == {'min_ns': 10.0, 'max_ns': 10.0, 'avg_ns': 10.0, 'samples': 2}
    assert latency['per_port']['y_p0.txt'] == latency['global']

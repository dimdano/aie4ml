"""Focused placement legality tests."""

import pytest
from aie4ml.op_impls.base import BufferLocation
from aie4ml.passes.placement import EdgeSpec, GraphSpec, Placed, PortFace, Rect, _placements_conflict


@pytest.mark.parametrize(
    ('row', 'producer_col', 'consumer_col'),
    [
        (0, 7, 8),  # Even AIE1 row: consumer input memory is west.
        (1, 8, 7),  # Odd AIE1 row: consumer input memory is east.
    ],
)
def test_direct_keepout_overlap_requires_exact_physical_bank_alias(row, producer_col, consumer_col):
    def output_locations(_row):
        return (BufferLocation('out', 0, 0, 0, (0, 3)),)

    def input_locations(anchor_row):
        return (BufferLocation('in', 0, -1 if anchor_row % 2 == 0 else 1, 0, (0, 3)),)

    producer = Placed(
        'producer',
        producer_col,
        row,
        Rect(
            1,
            1,
            PortFace('left', 0, 0),
            PortFace('right', 0, 0),
            buffer_locations=output_locations,
        ),
    )
    consumer = Placed(
        'consumer',
        consumer_col,
        row,
        Rect(
            1,
            1,
            PortFace('left', 0, 0),
            PortFace('right', 0, 0),
            keepout_left=1,
            keepout_right=1,
            buffer_locations=input_locations,
        ),
    )
    edge = EdgeSpec(
        'producer',
        'consumer',
        direct=True,
        src_group='out',
        dst_group='in',
        port_pairs=((0, 0),),
    )
    graph = GraphSpec([], {}, [edge], {}, {})

    assert not _placements_conflict(producer, consumer, graph)

    consumer.rect.buffer_locations = lambda anchor_row: (
        BufferLocation('in', 0, -1 if anchor_row % 2 == 0 else 1, 0, (1, 2)),
    )
    assert _placements_conflict(producer, consumer, graph)

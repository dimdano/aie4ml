from __future__ import annotations

from .families.matmul import DenseOpImplVariant


def register_builtin_op_impls(registry) -> None:
    registry.register(
        DenseOpImplVariant(
            variant_id='dense.b.r.v1',
            op_type='dense',
            supported_generations=('AIE-ML', 'AIE-MLV2'),
            supported_precisions=(
                {'input': 8, 'weight': 8, 'output': 8, 'acc': 32, 'bias': 32},
                {'input': 8, 'weight': 8, 'output': 16, 'acc': 32, 'bias': 32},
                {'input': 8, 'weight': 8, 'output': 32, 'acc': 32, 'bias': 32},
                {'input': 16, 'weight': 8, 'output': 8, 'acc': 32, 'bias': 32},
                {'input': 16, 'weight': 16, 'output': 16, 'acc': 64, 'bias': 32},
                {'input': 16, 'weight': 16, 'output': 32, 'acc': 64, 'bias': 32},
                {'input': 16, 'weight': 16, 'output': 16, 'acc': 32, 'bias': 32, 'input_c_type': 'bfloat16'},
                {'input': 32, 'weight': 32, 'output': 32, 'acc': 32, 'bias': 32, 'input_c_type': 'float'},
            ),
            supported_input_modes=('direct', 'memtile', 'plio', 'auto'),
            supported_output_modes=('direct', 'memtile', 'plio', 'auto'),
        )
    )

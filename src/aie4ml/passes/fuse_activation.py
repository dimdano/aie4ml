# Dense + Activation fusion pass for the AIE backend.

from hls4ml.model.optimizer.optimizer import ModelOptimizerPass
from ..ir import get_backend_context, LogicalIR, TraitInstance


class FuseActivationCasts(ModelOptimizerPass):
    """Fuse Dense+Activation pairs (relu or linear) directly in the hls4ml graph."""

    _SUPPORTED = {'relu', 'linear'}

    def __init__(self):
        self.name = 'fuse_activation_casts'

    def match(self, node):
        if getattr(node, 'op_type', None) != 'activation' or len(node.inputs) != 1:
            return False

        act = (node.metadata.get('activation', '')).lower()
        if act not in self._SUPPORTED:
            return False

        prev_node = node.inputs[0].producer
        if prev_node is None:
            return False
        match getattr(prev_node, 'op_type', None):
            case 'dense':
                return True
            case 'input':
                return act == 'linear'
            case _:
                return False
    
    def transform(self, model) -> bool:
        ctx = get_backend_context(model)
        graph: LogicalIR = ctx.ir.logical

        for i in range(len(graph.nodes) - 1, -1, -1):
            node = graph.nodes[i]
            if self.match(node):
                prev_node = node.inputs[0].producer
                activation = node.metadata.get('activation', '').lower()
                prev_node.add_trait(TraitInstance('fused_activation', {'activation': activation}))

                act_quant = node.metadata.get('quant', {})
                prev_quant = prev_node.metadata.get('quant', {})
                if not act_quant:
                    raise RuntimeError('Quantization information missing. Run quant before invoking downstream passes.')
                prev_quant['output_precision'] = act_quant['output_precision']

                graph.remove_node(node)

        return True
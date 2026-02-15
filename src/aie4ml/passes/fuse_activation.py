"""Dense+activation fusion pass on aie4ml logical IR."""

from hls4ml.model.optimizer.optimizer import ModelOptimizerPass

from ..ir import TraitInstance, get_backend_context


class FuseActivationCasts(ModelOptimizerPass):
    _SUPPORTED = {'relu', 'linear'}

    def __init__(self):
        self.name = 'fuse_activation_casts'

    def transform(self, model):
        ctx = get_backend_context(model)
        graph = ctx.ir.logical
        changed = False

        for act_node in list(graph.nodes):
            if act_node.op_type != 'activation' or len(act_node.inputs) != 1 or len(act_node.outputs) != 1:
                continue

            activation = (act_node.metadata.get('activation', '') or '').lower()
            if activation not in self._SUPPORTED:
                continue

            in_tensor = act_node.inputs[0]
            producer = in_tensor.producer
            if producer is None or producer.op_type != 'dense' or len(producer.outputs) != 1:
                continue

            producer.add_trait(TraitInstance('fused_activation', {'activation': activation}))

            act_quant = act_node.metadata.get('quant', {})
            output_precision = act_quant['output_precision']
            producer_quant = producer.metadata.setdefault('quant', {})
            producer_quant['output_precision'] = output_precision

            graph.remove_node(act_node, mode='contract')

            changed = True

        return changed

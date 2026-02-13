# Dense + Activation fusion pass for the AIE backend.

from hls4ml.model.optimizer import OptimizerPass


class FuseActivationCasts(OptimizerPass):
    """Fuse Dense+Activation pairs (relu or linear) directly in the hls4ml graph."""

    _SUPPORTED = {'relu', 'linear'}
    _FUSABLE = {'Dense', 'Conv1D', 'Conv2D'}

    def match(self, node):
        if getattr(node, 'class_name', None) != 'Activation' or len(node.inputs) != 1:
            return False

        act = (node.get_attr('activation', '') or '').lower()
        if act not in self._SUPPORTED:
            return False

        prev_node = node.get_input_node()
        if prev_node is None or getattr(prev_node, 'class_name', None) not in self._FUSABLE:
            return False

        return True

    def transform(self, model, node):
        dense = node.get_input_node()
        activation = (node.get_attr('activation', '') or '').lower()

        in_var = node.get_input_variable()
        out_var = node.get_output_variable()
        in_var.type.precision = out_var.type.precision

        dense.set_attr('aie_fused_activation', activation)
        model.remove_node(node)
        return True

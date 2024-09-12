import tempfile
from pathlib import Path

import openvino
from transformers import AutoTokenizer
from transformers.onnx import export
from transformers.onnx.features import FeaturesManager


class ModelLoader:
    @classmethod
    def load(cls, model_name: str) -> tuple[openvino.CompiledModel, AutoTokenizer]:
        # load model
        model = FeaturesManager.get_model_from_feature("default", model_name)

        # check model config
        _, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature="default")
        onnx_config = model_onnx_config(model.config)

        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # load openvino model
        core = openvino.Core()
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / f"{model_name}.onnx"
            export(tokenizer, model, onnx_config, onnx_config.default_onnx_opset, onnx_path)

            # read .onnx with openvino
            openvino_model = core.read_model(onnx_path)

        # Enforce dynamic input shape
        try:
            openvino_model.reshape({
                input_.any_name: openvino.PartialShape([1, "?"]) for input_ in openvino_model.inputs
            })
        except RuntimeError:
            raise

        throughput = {'PERFORMANCE_HINT': 'THROUGHPUT'}
        compiled_model = core.compile_model(openvino_model, "CPU", throughput)
        #compiled_model = core.compile_model(openvino_model, "AUTO")
        
        return compiled_model, tokenizer

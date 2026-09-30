"""ONNX Runtime session helper. Uses the Qualcomm NPU (QNN HTP) when available."""
import onnxruntime as ort


def available_providers():
    return ort.get_available_providers()


def make_session(model_path, prefer="qnn", backend_path="QnnHtp.dll"):
    """Create an InferenceSession, requesting the NPU first and falling back to CPU."""
    providers = []
    if prefer == "qnn" and "QNNExecutionProvider" in available_providers():
        providers.append(
            (
                "QNNExecutionProvider",
                {"backend_path": backend_path, "htp_performance_mode": "burst"},
            )
        )
    providers.append("CPUExecutionProvider")
    return ort.InferenceSession(str(model_path), providers=providers)

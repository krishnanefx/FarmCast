import numpy as np
import onnxruntime as ort
import os
from dotenv import load_dotenv

# Load model paths from .env
load_dotenv()

MODEL_PATH = os.getenv("PANGU_MODEL_1H_PATH", "models/pangu_weather_1.onnx")  # change if needed

def main():
    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    # Pangu input shapes from your earlier inspection
    upper = np.zeros((5, 13, 721, 1440), dtype=np.float32)
    surface = np.zeros((4, 721, 1440), dtype=np.float32)

    out_upper, out_surface = session.run(
        None,
        {
            "input": upper,
            "input_surface": surface,
        },
    )

    print("upper in :", upper.shape, upper.dtype)
    print("surf  in :", surface.shape, surface.dtype)
    print("upper out:", out_upper.shape, out_upper.dtype)
    print("surf  out:", out_surface.shape, out_surface.dtype)
    print("upper out stats:", float(out_upper.min()), float(out_upper.max()), float(out_upper.mean()))
    print("surf  out stats:", float(out_surface.min()), float(out_surface.max()), float(out_surface.mean()))

if __name__ == "__main__":
    main()
import argparse
import time
import numpy as np
import onnxruntime as ort


def parse_args():
    parser = argparse.ArgumentParser(description="ONNXRuntime latency benchmark for FastDepth.")
    parser.add_argument("--model", required=True, type=str, help="ONNX model path")
    parser.add_argument("--height", default=512, type=int)
    parser.add_argument("--width", default=768, type=int)
    parser.add_argument("--runs", default=100, type=int)
    parser.add_argument("--warmup", default=20, type=int)
    parser.add_argument("--provider", default="CPUExecutionProvider", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    sess = ort.InferenceSession(args.model, providers=[args.provider])
    input_name = sess.get_inputs()[0].name
    x = np.random.rand(1, 3, args.height, args.width).astype(np.float32)

    for _ in range(args.warmup):
        sess.run(None, {input_name: x})

    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: x})
        times.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = float(np.mean(times))
    p95_ms = float(np.percentile(times, 95))
    print(f"Provider={args.provider}")
    print(f"Runs={args.runs}, Warmup={args.warmup}")
    print(f"AVG={avg_ms:.3f} ms, P95={p95_ms:.3f} ms, FPS={1000.0 / avg_ms:.2f}")


if __name__ == "__main__":
    main()

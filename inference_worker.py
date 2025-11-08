#!/usr/bin/env python3

"""
MapAnything inference worker (Python 3.12).

Listens on a ZeroMQ socket for image batches from the ROS node, runs inference
using the MapAnything model, and sends predictions back.
"""

import os
import pickle
import signal
import sys
from typing import List, Dict, Any

import numpy as np
import torch
import zmq

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs


class MapAnythingInferenceWorker:
    def __init__(self, zmq_endpoint: str = "ipc:///tmp/mapanything.ipc"):
        self.zmq_endpoint = zmq_endpoint
        self.context = None
        self.socket = None
        self.running = True

        # Load model
        print("Loading MapAnything model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MapAnything.from_pretrained("facebook/map-anything").to(device)
        self.model.eval()
        print(f"Model loaded on device: {device}")

        # Setup ZeroMQ
        self.setup_zmq()

        # Setup signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def setup_zmq(self):
        """Setup ZeroMQ REP socket for communication."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(self.zmq_endpoint)
        print(f"Inference worker listening on {self.zmq_endpoint}")

    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\nReceived signal {signum}, shutting down...")
        self.running = False
        sys.exit(0)

    def run(self):
        """Main loop: receive and process requests."""
        print("Waiting for requests from ROS node...")

        while self.running:
            try:
                # Receive request (blocking)
                message_data = self.socket.recv()

                print(f"Received message ({len(message_data)} bytes)")

                # Deserialize batch data
                batch_data = pickle.loads(message_data)
                images = batch_data["images"]
                intrinsics = batch_data["intrinsics"]

                print(f"Processing batch of {len(images)} images")

                # Run inference
                response = self.process_batch(images, intrinsics)

                # Serialize response
                serialized_response = pickle.dumps(response)

                # Send response
                self.socket.send(serialized_response)

                print(f"Sent response ({len(serialized_response)} bytes)")

            except zmq.ZMQError as exc:
                if self.running:
                    print(f"ZMQ error: {exc}")
                break
            except Exception as exc:
                print(f"Error processing request: {exc}")
                import traceback

                traceback.print_exc()

                # Send error response
                try:
                    error_response = {
                        "status": "error",
                        "error": str(exc),
                    }
                    serialized_error = pickle.dumps(error_response)
                    self.socket.send(serialized_error)
                except Exception as send_exc:
                    print(f"Failed to send error response: {send_exc}")

    def process_batch(
        self, images: List[np.ndarray], intrinsics: np.ndarray
    ) -> Dict[str, Any]:
        """Run inference on a batch of images."""
        try:
            # Prepare views for the model
            views = [{"img": image, "intrinsics": intrinsics} for image in images]

            # Preprocess inputs
            processed_views = preprocess_inputs(views)

            # Run inference
            with torch.no_grad():
                predictions = self.model.infer(
                    processed_views,
                    memory_efficient_inference=False,
                    use_amp=True,
                    amp_dtype="bf16",
                    apply_mask=True,
                    mask_edges=True,
                    apply_confidence_mask=False,
                    confidence_percentile=10,
                )

            print(f"Inference completed successfully")

            # Return success response with predictions
            # Note: Convert tensors to numpy for serialization
            serializable_predictions = self.make_serializable(predictions)

            return {
                "status": "success",
                "predictions": serializable_predictions,
            }

        except Exception as exc:
            print(f"Inference error: {exc}")
            import traceback

            traceback.print_exc()

            return {
                "status": "error",
                "error": str(exc),
            }

    def make_serializable(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Convert model predictions to serializable format (numpy arrays)."""
        serializable = {}

        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                # Move to CPU and convert to numpy
                serializable[key] = value.cpu().numpy()
            elif isinstance(value, dict):
                # Recursively process nested dicts
                serializable[key] = self.make_serializable(value)
            elif isinstance(value, list):
                # Process lists
                serializable[key] = [
                    item.cpu().numpy() if isinstance(item, torch.Tensor) else item
                    for item in value
                ]
            else:
                # Keep as is
                serializable[key] = value

        return serializable

    def cleanup(self):
        """Clean up resources."""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        print("Cleanup complete")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="MapAnything inference worker (Python 3.12)"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="ipc:///tmp/mapanything.ipc",
        help="ZeroMQ endpoint to bind (e.g., tcp://*:5555 or ipc:///tmp/mapanything.ipc)",
    )
    args = parser.parse_args()

    worker = MapAnythingInferenceWorker(zmq_endpoint=args.endpoint)

    try:
        worker.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        worker.cleanup()


if __name__ == "__main__":
    main()

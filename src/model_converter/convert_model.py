import torch
import pathlib
import sys
import os
import pickle
import io

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from happy.models.clustergcn import ClusterGCN

def load_legacy_model(model_path, map_location="cpu"):
    """Load a model saved with an older version of PyTorch."""
    with open(model_path, 'rb') as f:
        try:
            # Try direct loading first
            return torch.load(f, map_location=map_location)
        except Exception as e:
            print(f"Direct loading failed: {e}")
            print("Attempting to load state dict only...")
            # Reset file pointer
            f.seek(0)
            # Load with pickle to get the state dict
            buffer = io.BytesIO(f.read())
            buffer.seek(0)
            unpickler = pickle.Unpickler(buffer)
            state_dict = None
            try:
                while True:
                    obj = unpickler.load()
                    if isinstance(obj, dict) and '_metadata' in obj:
                        state_dict = obj
                        break
            except EOFError:
                pass

            if state_dict is None:
                raise ValueError("Could not find state dict in model file")

            # Create a new model instance
            model = ClusterGCN()
            # Load the state dict
            model.load_state_dict(state_dict)
            return model

def main(model_dir):
    model_path = pathlib.Path(f"{model_dir}/graph_model.pt")  # Change if needed
    output_path = pathlib.Path(f"{model_dir}/graph_converted_state_dict.pth")

    # Load full model (only works in PyTorch 2.0.1 with matching class definition)
    print(f"Loading full model from: {model_path}")
    model = load_legacy_model(model_path, map_location="cpu")
    print(model)
    # Save only the state_dict (forward-compatible)
    print(f"Saving state_dict to: {output_path}")
    torch.save(model.state_dict(), output_path)

    print("✅ Conversion complete: state_dict saved.")

if __name__ == "__main__":
    model_dir = "/mnt/run/jh/tools/happy/projects/placenta/trained_models"
    main(model_dir)
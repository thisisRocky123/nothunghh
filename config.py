from pathlib import Path

IMG_SIZE = 28
N_FEATURES = 4
N_QUBITS = 4
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATHS = [BASE_DIR / "model.pth", BASE_DIR / "best_model.pt", BASE_DIR / "hqnn_checkpoint.pth"]
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

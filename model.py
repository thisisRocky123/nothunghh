import numpy as np
import torch
import torch.nn as nn

from config import IMG_SIZE, N_FEATURES, N_QUBITS


def _single_qubit_gate(theta, gate_type):
    if gate_type == "rx":
        c = np.cos(theta / 2.0)
        s = np.sin(theta / 2.0)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
    if gate_type == "ry":
        c = np.cos(theta / 2.0)
        s = np.sin(theta / 2.0)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    if gate_type == "rz":
        return np.array([[np.exp(-1j * theta / 2.0), 0.0], [0.0, np.exp(1j * theta / 2.0)]], dtype=np.complex128)
    raise ValueError(f"Unsupported gate type: {gate_type}")


def _apply_single_qubit_gate(state, gate, target, n_qubits):
    new_state = np.zeros_like(state)
    for idx in range(2**n_qubits):
        if ((idx >> target) & 1) == 0:
            j = idx | (1 << target)
            a0 = state[idx]
            a1 = state[j]
            new_state[idx] = gate[0, 0] * a0 + gate[0, 1] * a1
            new_state[j] = gate[1, 0] * a0 + gate[1, 1] * a1
        else:
            new_state[idx] = state[idx]
    return new_state


def _apply_cnot(state, control, target, n_qubits):
    new_state = np.zeros_like(state)
    for idx in range(2**n_qubits):
        if ((idx >> control) & 1) == 1:
            j = idx ^ (1 << target)
            new_state[j] = state[idx]
        else:
            new_state[idx] = state[idx]
    return new_state


def _run_quantum_circuit(theta_vals, alpha_vals, phi_vals, lam_vals):
    n_qubits = len(theta_vals)
    state = np.zeros(2**n_qubits, dtype=np.complex128)
    state[0] = 1.0

    for q in range(n_qubits):
        state = _apply_single_qubit_gate(state, _single_qubit_gate(theta_vals[q], "ry"), q, n_qubits)
        state = _apply_single_qubit_gate(state, _single_qubit_gate(alpha_vals[q], "rx"), q, n_qubits)
        state = _apply_single_qubit_gate(state, _single_qubit_gate(phi_vals[q], "rz"), q, n_qubits)
        state = _apply_single_qubit_gate(state, _single_qubit_gate(lam_vals[q], "ry"), q, n_qubits)

    for q in range(n_qubits - 1):
        state = _apply_cnot(state, q, q + 1, n_qubits)
    for q in range(n_qubits - 1, 0, -1):
        state = _apply_cnot(state, q, q - 1, n_qubits)

    probs = np.abs(state) ** 2
    exp_vals = np.zeros(n_qubits, dtype=np.float32)
    for q in range(n_qubits):
        p0 = probs[[idx for idx in range(2**n_qubits) if ((idx >> q) & 1) == 0]].sum()
        p1 = probs[[idx for idx in range(2**n_qubits) if ((idx >> q) & 1) == 1]].sum()
        exp_vals[q] = p0 - p1
    return exp_vals


class CNNFeatureExtractor(nn.Module):
    def __init__(self, n_features=4, img_size=28):
        super().__init__()
        self.n_features = n_features
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        flat_size = 64 * (img_size // 8) * (img_size // 8)
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_features),
        )
        self.angle_scale = nn.Parameter(torch.tensor(np.pi), requires_grad=False)

    def forward(self, x):
        print("CNN block 1 start")
        x = self.conv_block(x)
        print("CNN block 1 done")
        print("CNN block 2 start")
        x = self.fc_block(x)
        print("CNN block 2 done")
        print("CNN block 3 start")
        x = torch.tanh(x) * self.angle_scale
        print("CNN block 3 done")
        return x


class SingleStagePQCLayer(nn.Module):
    def __init__(self, n_qubits=4):
        super().__init__()
        self.n_qubits = n_qubits
        self.alpha_weights = nn.Parameter(torch.zeros(n_qubits))
        self.phi_weights = nn.Parameter(torch.zeros(n_qubits))
        self.lam_weights = nn.Parameter(torch.zeros(n_qubits))

    def forward(self, theta_tensor):
        batch_size = theta_tensor.shape[0]
        outputs = []
        for b in range(batch_size):
            theta_vals = theta_tensor[b].detach().cpu().numpy()
            alpha_vals = self.alpha_weights.detach().cpu().numpy()
            phi_vals = self.phi_weights.detach().cpu().numpy()
            lam_vals = self.lam_weights.detach().cpu().numpy()
            outputs.append(_run_quantum_circuit(theta_vals, alpha_vals, phi_vals, lam_vals))
        return torch.tensor(np.stack(outputs), dtype=torch.float32, device=theta_tensor.device)


class HQNNModel(nn.Module):
    def __init__(self, n_features=N_FEATURES, n_qubits=N_QUBITS, img_size=IMG_SIZE):
        super().__init__()
        self.cnn = CNNFeatureExtractor(n_features=n_features, img_size=img_size)
        self.quantum = SingleStagePQCLayer(n_qubits=n_qubits)
        self.classical_head = nn.Sequential(nn.Linear(n_qubits, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print("1 CNN start")
        features = self.cnn(x)
        print("2 CNN done")
        print("3 Quantum start")
        z_vals = self.quantum(features)
        print("4 Quantum done")
        print("5 Classifier start")
        logit = self.classical_head(z_vals)
        prob = self.sigmoid(logit)
        print("6 Classifier done")
        return prob.squeeze(1)


def build_model(checkpoint_path=None, device="cpu"):
    model = HQNNModel()
    model.to(device)
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=True)
    return model


def load_model(checkpoint_path=None, device="cpu"):
    if checkpoint_path is None:
        from config import CHECKPOINT_PATHS

        for candidate in CHECKPOINT_PATHS:
            if candidate.exists():
                checkpoint_path = str(candidate)
                break
        if checkpoint_path is None:
            raise FileNotFoundError("No checkpoint file found. Expected model.pth, best_model.pt, or hqnn_checkpoint.pth")
    model = build_model(checkpoint_path=checkpoint_path, device=device)
    model.eval()
    return model

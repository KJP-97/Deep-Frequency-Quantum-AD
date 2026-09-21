# ================================================================
# Hybrid Deep-Frequency Quantum Framework
# Vision Mamba -> AQBPSO -> VQC -> Attention Fusion
#
# Input:
#   Hybrid features extracted from:
#   EfficientNet-B4 + Radiomics + WPT + LBP
# ================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Quantum libraries
import pennylane as qml


# ================================================================
# 1. Configuration
# ================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_CLASSES = 4

# Vision Mamba parameters
MAMBA_DIM = 256
MAMBA_LAYERS = 2

# AQBPSO
AQBPSO_PARTICLES = 20
AQBPSO_ITERATIONS = 30
AQBPSO_SELECTED_FEATURES = 1024

# Quantum classifier
N_QUBITS = 10
VQC_LAYERS = 2

# Final feature dimension
FUSION_DIM = 128


# ================================================================
# 2. Load Hybrid Features
# ================================================================

def load_hybrid_features(
    feature_file,
    label_file
):

    X = np.load(feature_file)
    y = np.load(label_file)

    print("Hybrid feature matrix:", X.shape)
    print("Labels:", y.shape)

    return X, y


# ================================================================
# 3. Feature Normalization
# ================================================================

def normalize_features(X_train, X_test):

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_test = scaler.transform(
        X_test
    )

    return X_train, X_test, scaler


# ================================================================
# 4. Vision Mamba Feature Encoder
# ================================================================

class VisionMambaBlock(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=256
    ):

        super().__init__()

        self.input_projection = nn.Linear(
            input_dim,
            hidden_dim
        )

        # Sequence modelling layer.
        #
        # This provides the sequence-processing interface
        # used before the Mamba-style state-space representation.
        self.sequence_model = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.norm = nn.LayerNorm(
            hidden_dim
        )

    def forward(self, x):

        x = self.input_projection(x)

        # Convert feature vector into sequence
        x = x.unsqueeze(1)

        residual = x

        x = self.sequence_model(x)

        x = x + residual

        x = self.norm(x)

        # Global sequence representation
        x = torch.mean(
            x,
            dim=1
        )

        return x


class VisionMambaEncoder(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=MAMBA_DIM
    ):

        super().__init__()

        self.encoder = VisionMambaBlock(
            input_dim,
            hidden_dim
        )

        self.projection = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x):

        x = self.encoder(x)

        x = self.projection(x)

        return x


# ================================================================
# 5. AQBPSO Feature Selection
# ================================================================

class AQBPSO:

    def __init__(
        self,
        num_features,
        num_particles=AQBPSO_PARTICLES,
        max_iterations=AQBPSO_ITERATIONS,
        selected_features=AQBPSO_SELECTED_FEATURES
    ):

        self.num_features = num_features
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.selected_features = min(
            selected_features,
            num_features
        )

        # Binary population
        self.population = np.random.randint(
            0,
            2,
            size=(
                num_particles,
                num_features
            )
        )

        # Guarantee minimum number of selected features
        for i in range(num_particles):

            if np.sum(
                self.population[i]
            ) == 0:

                indices = np.random.choice(
                    num_features,
                    self.selected_features,
                    replace=False
                )

                self.population[
                    i,
                    indices
                ] = 1

    # ------------------------------------------------------------
    # Fitness Function
    # ------------------------------------------------------------

    def fitness(
        self,
        feature_mask,
        X,
        y
    ):

        selected = np.where(
            feature_mask == 1
        )[0]

        if len(selected) == 0:
            return 1e10

        # Variance-based discriminative proxy
        selected_features = X[:, selected]

        feature_variance = np.mean(
            np.var(
                selected_features,
                axis=0
            )
        )

        # Feature-selection penalty
        selection_ratio = (
            len(selected) /
            self.num_features
        )

        score = (
            -feature_variance
            + 0.01 * selection_ratio
        )

        return score

    # ------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------

    def optimize(
        self,
        X,
        y
    ):

        best_mask = None
        best_score = np.inf

        for iteration in range(
            self.max_iterations
        ):

            scores = []

            for i in range(
                self.num_particles
            ):

                score = self.fitness(
                    self.population[i],
                    X,
                    y
                )

                scores.append(score)

                if score < best_score:

                    best_score = score

                    best_mask = (
                        self.population[i]
                        .copy()
                    )

            # Adaptive binary update
            for i in range(
                self.num_particles
            ):

                probability = (
                    1.0 -
                    iteration /
                    self.max_iterations
                )

                random_values = np.random.rand(
                    self.num_features
                )

                self.population[i] = np.where(
                    random_values < probability,
                    self.population[i],
                    np.random.randint(
                        0,
                        2,
                        self.num_features
                    )
                )

            print(
                f"AQBPSO iteration "
                f"{iteration + 1}/"
                f"{self.max_iterations} | "
                f"Best fitness: "
                f"{best_score:.6f}"
            )

        selected_indices = np.where(
            best_mask == 1
        )[0]

        # Keep the strongest features if
        # more than the requested number remain.
        if len(selected_indices) > self.selected_features:

            variances = np.var(
                X[:, selected_indices],
                axis=0
            )

            ranking = np.argsort(
                variances
            )[::-1]

            selected_indices = (
                selected_indices[
                    ranking[
                        :self.selected_features
                    ]
                ]
            )

        print(
            "\nAQBPSO selected features:",
            len(selected_indices)
        )

        return selected_indices


# ================================================================
# 6. Dimension Reduction Before Quantum Encoding
# ================================================================

class QuantumFeatureReducer(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim=N_QUBITS
    ):

        super().__init__()

        self.reducer = nn.Sequential(

            nn.Linear(
                input_dim,
                256
            ),

            nn.GELU(),

            nn.Linear(
                256,
                128
            ),

            nn.GELU(),

            nn.Linear(
                128,
                output_dim
            ),

            nn.Tanh()
        )

    def forward(self, x):

        return self.reducer(x)


# ================================================================
# 7. Variational Quantum Circuit
# ================================================================

dev = qml.device(
    "default.qubit",
    wires=N_QUBITS
)


@qml.qnode(
    dev,
    interface="torch"
)
def quantum_circuit(
    inputs,
    weights
):

    # ------------------------------------------------------------
    # Angle encoding
    # ------------------------------------------------------------

    for i in range(N_QUBITS):

        qml.RY(
            inputs[i],
            wires=i
        )

        qml.RZ(
            inputs[i],
            wires=i
        )

    # ------------------------------------------------------------
    # Variational layers
    # ------------------------------------------------------------

    for layer in range(
        VQC_LAYERS
    ):

        for qubit in range(
            N_QUBITS
        ):

            qml.RX(
                weights[
                    layer,
                    qubit,
                    0
                ],
                wires=qubit
            )

            qml.RY(
                weights[
                    layer,
                    qubit,
                    1
                ],
                wires=qubit
            )

            qml.RZ(
                weights[
                    layer,
                    qubit,
                    2
                ],
                wires=qubit
            )

        # Entanglement
        for qubit in range(
            N_QUBITS - 1
        ):

            qml.CNOT(
                wires=[
                    qubit,
                    qubit + 1
                ]
            )

    return [
        qml.expval(
            qml.PauliZ(i)
        )
        for i in range(N_QUBITS)
    ]


# ================================================================
# 8. VQC Layer
# ================================================================

class VQCLayer(nn.Module):

    def __init__(
        self,
        n_qubits=N_QUBITS
    ):

        super().__init__()

        self.weights = nn.Parameter(
            0.01 * torch.randn(
                VQC_LAYERS,
                n_qubits,
                3
            )
        )

    def forward(self, x):

        outputs = []

        for sample in x:

            result = quantum_circuit(
                sample,
                self.weights
            )

            result = torch.stack(
                result
            )

            outputs.append(result)

        return torch.stack(
            outputs
        )


# ================================================================
# 9. Attention Fusion
# ================================================================

class AttentionFusion(nn.Module):

    def __init__(
        self,
        feature_dim,
        fusion_dim=FUSION_DIM
    ):

        super().__init__()

        self.projection = nn.Linear(
            feature_dim,
            fusion_dim
        )

        self.attention = nn.Sequential(

            nn.Linear(
                fusion_dim,
                fusion_dim // 2
            ),

            nn.Tanh(),

            nn.Linear(
                fusion_dim // 2,
                1
            )
        )

        self.output = nn.Sequential(

            nn.Linear(
                fusion_dim,
                fusion_dim
            ),

            nn.GELU(),

            nn.LayerNorm(
                fusion_dim
            )
        )

    def forward(self, x):

        x = self.projection(x)

        attention_scores = self.attention(
            x
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=0
        )

        fused = (
            x *
            attention_weights
        )

        fused = self.output(
            fused
        )

        return fused


# ================================================================
# 10. Complete Classification Model
# ================================================================

class HybridQuantumADModel(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        selected_dim=AQBPSO_SELECTED_FEATURES
    ):

        super().__init__()

        # Vision Mamba
        self.mamba = VisionMambaEncoder(
            input_dim=input_dim,
            hidden_dim=MAMBA_DIM
        )

        # AQBPSO output projection
        self.aqbpso_projection = nn.Linear(
            MAMBA_DIM,
            selected_dim
        )

        # Quantum feature reduction
        self.quantum_reducer = (
            QuantumFeatureReducer(
                selected_dim,
                N_QUBITS
            )
        )

        # VQC
        self.vqc = VQCLayer(
            N_QUBITS
        )

        # Attention Fusion
        self.attention_fusion = (
            AttentionFusion(
                N_QUBITS,
                FUSION_DIM
            )
        )

        # Final classifier
        self.classifier = nn.Sequential(

            nn.Linear(
                FUSION_DIM,
                64
            ),

            nn.GELU(),

            nn.Dropout(0.3),

            nn.Linear(
                64,
                NUM_CLASSES
            )
        )

    def forward(self, x):

        # --------------------------------------------------------
        # Vision Mamba
        # --------------------------------------------------------

        mamba_features = self.mamba(x)

        # --------------------------------------------------------
        # Feature representation for AQBPSO
        # --------------------------------------------------------

        optimized_space = (
            self.aqbpso_projection(
                mamba_features
            )
        )

        # --------------------------------------------------------
        # Quantum dimensionality reduction
        # --------------------------------------------------------

        quantum_input = (
            self.quantum_reducer(
                optimized_space
            )
        )

        # --------------------------------------------------------
        # VQC
        # --------------------------------------------------------

        quantum_features = self.vqc(
            quantum_input
        )

        # --------------------------------------------------------
        # Attention Fusion
        # --------------------------------------------------------

        fused_features = (
            self.attention_fusion(
                quantum_features
            )
        )

        # --------------------------------------------------------
        # Classification
        # --------------------------------------------------------

        logits = self.classifier(
            fused_features
        )

        return {
            "logits": logits,
            "mamba_features": mamba_features,
            "optimized_features": optimized_space,
            "quantum_features": quantum_features,
            "fused_features": fused_features
        }


# ================================================================
# 11. Example Pipeline
# ================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------
    # Load hybrid features
    # ------------------------------------------------------------

    X = np.load(
        "results/hybrid_features.npy"
    )

    y = np.load(
        "results/labels.npy"
    )

    print(
        "\nInput hybrid features:",
        X.shape
    )

    # ------------------------------------------------------------
    # Train-test split
    # ------------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=42
        )
    )

    # ------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------

    X_train, X_test, scaler = (
        normalize_features(
            X_train,
            X_test
        )
    )

    # ------------------------------------------------------------
    # AQBPSO feature selection
    # ------------------------------------------------------------

    aqbpso = AQBPSO(
        num_features=X_train.shape[1],
        num_particles=20,
        max_iterations=30,
        selected_features=1024
    )

    selected_indices = aqbpso.optimize(
        X_train,
        y_train
    )

    X_train_selected = (
        X_train[:, selected_indices]
    )

    X_test_selected = (
        X_test[:, selected_indices]
    )

    print(
        "After AQBPSO:",
        X_train_selected.shape
    )

    # ------------------------------------------------------------
    # Convert to PyTorch
    # ------------------------------------------------------------

    X_train_tensor = torch.tensor(
        X_train_selected,
        dtype=torch.float32
    ).to(DEVICE)

    X_test_tensor = torch.tensor(
        X_test_selected,
        dtype=torch.float32
    ).to(DEVICE)

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.long
    ).to(DEVICE)

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    model = HybridQuantumADModel(
        input_dim=X_train_selected.shape[1],
        selected_dim=X_train_selected.shape[1]
    ).to(DEVICE)

    # ------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------

    output = model(
        X_train_tensor[:4]
    )

    print(
        "\nClassification output:",
        output["logits"].shape
    )

    print(
        "Vision Mamba output:",
        output["mamba_features"].shape
    )

    print(
        "Quantum output:",
        output["quantum_features"].shape
    )

    print(
        "Attention-fused output:",
        output["fused_features"].shape
    )

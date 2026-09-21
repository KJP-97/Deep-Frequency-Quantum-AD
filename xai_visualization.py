# ================================================================
# Explainability and Visualization
# Eigen-CAM + t-SNE
# ================================================================

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from torchvision import models, transforms


# ================================================================
# Configuration
# ================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 224

CLASS_NAMES = [
    "Mild",
    "Moderate",
    "No Impairment",
    "Very Mild"
]

OUTPUT_DIR = "results/explainability"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ================================================================
# 1. EfficientNet-B4 Model
# ================================================================

def load_efficientnet():

    model = models.efficientnet_b4(
        weights=models.EfficientNet_B4_Weights.DEFAULT
    )

    model = model.to(DEVICE)
    model.eval()

    return model


# ================================================================
# 2. Image Preprocessing
# ================================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.Grayscale(
        num_output_channels=3
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ================================================================
# 3. Eigen-CAM
# ================================================================

class EigenCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model
        self.target_layer = target_layer

        self.activations = None

        self.hook = target_layer.register_forward_hook(
            self.save_activation
        )

    def save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output.detach()

    def generate(
        self,
        image_tensor
    ):

        self.model.zero_grad()

        with torch.no_grad():

            output = self.model(
                image_tensor
            )

        # --------------------------------------------------------
        # Feature activation
        # --------------------------------------------------------

        activation = self.activations

        # Expected shape:
        # [Batch, Channels, Height, Width]

        if activation.ndim != 4:

            raise ValueError(
                "Target layer must produce "
                "a 4-D feature map."
            )

        activation = activation[0]

        # --------------------------------------------------------
        # Principal component analysis
        # --------------------------------------------------------

        channels, height, width = (
            activation.shape
        )

        features = activation.reshape(
            channels,
            height * width
        )

        # Center features
        features = (
            features -
            features.mean(
                dim=1,
                keepdim=True
            )
        )

        # Covariance matrix
        covariance = torch.matmul(
            features,
            features.T
        )

        # Eigen decomposition
        eigenvalues, eigenvectors = (
            torch.linalg.eigh(
                covariance
            )
        )

        # Principal eigenvector
        principal_vector = (
            eigenvectors[:, -1]
        )

        # Eigen-CAM activation map
        cam = torch.matmul(
            principal_vector,
            features
        )

        cam = cam.reshape(
            height,
            width
        )

        # ReLU
        cam = F.relu(cam)

        # Normalize
        cam -= cam.min()

        if cam.max() > 0:

            cam /= cam.max()

        # Resize to input image size
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().cpu().numpy()

        prediction = torch.argmax(
            output,
            dim=1
        ).item()

        return cam, prediction

    def close(self):

        self.hook.remove()


# ================================================================
# 4. Overlay Eigen-CAM on MRI
# ================================================================

def overlay_eigen_cam(
    image_path,
    cam,
    prediction,
    output_path
):

    original = Image.open(
        image_path
    ).convert("RGB")

    original = original.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    original = np.asarray(
        original
    ).astype(np.float32) / 255.0

    # CAM heatmap
    plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(
        original
    )

    plt.imshow(
        cam,
        cmap="jet",
        alpha=0.45
    )

    plt.title(
        f"Eigen-CAM: "
        f"{CLASS_NAMES[prediction]}"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ================================================================
# 5. Run Eigen-CAM
# ================================================================

def run_eigen_cam(
    image_path,
    output_name="eigen_cam.png"
):

    model = load_efficientnet()

    # Final convolutional feature layer
    target_layer = (
        model.features[-1]
    )

    image = Image.open(
        image_path
    ).convert("L")

    tensor = transform(
        image
    ).unsqueeze(0)

    tensor = tensor.to(
        DEVICE
    )

    cam_generator = EigenCAM(
        model,
        target_layer
    )

    cam, prediction = (
        cam_generator.generate(
            tensor
        )
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    overlay_eigen_cam(
        image_path,
        cam,
        prediction,
        output_path
    )

    cam_generator.close()

    print(
        "Eigen-CAM saved to:",
        output_path
    )

    print(
        "Predicted class:",
        CLASS_NAMES[prediction]
    )


# ================================================================
# 6. t-SNE Visualization
# ================================================================

def tsne_visualization(
    features,
    labels,
    output_name="tsne_visualization.png"
):

    print(
        "\nRunning t-SNE..."
    )

    features = np.asarray(
        features,
        dtype=np.float32
    )

    labels = np.asarray(
        labels
    )

    # ------------------------------------------------------------
    # Standardization
    # ------------------------------------------------------------

    scaler = StandardScaler()

    features = scaler.fit_transform(
        features
    )

    # ------------------------------------------------------------
    # t-SNE
    # ------------------------------------------------------------

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        max_iter=1000,
        random_state=42
    )

    embedded = tsne.fit_transform(
        features
    )

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------

    plt.figure(
        figsize=(9, 7)
    )

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        indices = (
            labels == class_id
        )

        plt.scatter(
            embedded[
                indices,
                0
            ],
            embedded[
                indices,
                1
            ],
            label=class_name,
            alpha=0.70,
            s=25
        )

    plt.xlabel(
        "t-SNE Dimension 1"
    )

    plt.ylabel(
        "t-SNE Dimension 2"
    )

    plt.title(
        "t-SNE Visualization of Learned Features"
    )

    plt.legend()

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Save numerical t-SNE representation
    np.save(
        os.path.join(
            OUTPUT_DIR,
            "tsne_embeddings.npy"
        ),
        embedded
    )

    print(
        "t-SNE visualization saved to:",
        output_path
    )

    return embedded


# ================================================================
# 7. Compare Feature Representations
# ================================================================

def compare_feature_spaces(
    original_features,
    optimized_features,
    labels
):

    """
    Generate t-SNE representations for:

    1. Original hybrid features
    2. AQBPSO-optimized features
    """

    print(
        "\nVisualizing original hybrid features..."
    )

    tsne_visualization(
        original_features,
        labels,
        "tsne_original_features.png"
    )

    print(
        "\nVisualizing AQBPSO optimized features..."
    )

    tsne_visualization(
        optimized_features,
        labels,
        "tsne_aqbpso_features.png"
    )


# ================================================================
# 8. Example Usage
# ================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------
    # Eigen-CAM example
    # ------------------------------------------------------------

    IMAGE_PATH = (
        "data/sample_mri.jpg"
    )

    if Path(
        IMAGE_PATH
    ).exists():

        run_eigen_cam(
            IMAGE_PATH,
            "eigen_cam_sample.png"
        )

    # ------------------------------------------------------------
    # t-SNE 
    # ------------------------------------------------------------

    FEATURE_FILE = (
        "results/hybrid_features.npy"
    )

    LABEL_FILE = (
        "results/labels.npy"
    )

    if (
        Path(FEATURE_FILE).exists()
        and
        Path(LABEL_FILE).exists()
    ):

        features = np.load(
            FEATURE_FILE
        )

        labels = np.load(
            LABEL_FILE
        )

        tsne_visualization(
            features,
            labels
        )

    print(
        "\nXAI and visualization processing completed."
    )

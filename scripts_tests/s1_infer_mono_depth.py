"""
Offline depth estimation using DepthAnythingV2 over a batch of RGB images.

This module wraps the DepthBatchInferencer class for inference from a
predefined RGB dataset directory, saving depth outputs to disk.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.inference.depth_batch_inferencer import DepthBatchInferencer


class DepthOfflineInference:
    """
    Handles monocular depth estimation from a folder of RGB images using
    the DepthAnythingV2 model. Saves results in PNG and NPY formats.
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        checkpoint_dir: Path,
        encoder: str = "vits"
    ) -> None:
        """
        Initializes the offline inference pipeline.

        Args:
            input_dir (Path): Directory containing RGB input images.
            output_dir (Path): Output directory for depth maps.
            checkpoint_dir (Path): Directory with model checkpoint files.
            encoder (str): Encoder type ('vits', 'vitb', 'vitl', 'vitg').
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        self.encoder = encoder.lower()

        self._validate_paths()
        self.inferencer = DepthBatchInferencer(
            input_dir=self.input_dir,
            output_path=self.output_dir,
            encoder=self.encoder,
            checkpoint_dir=self.checkpoint_dir
        )

    def _validate_paths(self) -> None:
        """
        Validates the existence of required directories.
        Raises:
            FileNotFoundError: If input or checkpoint directory is missing.
        """
        if not self.input_dir.exists():
            raise FileNotFoundError(
                f"[ERROR] RGB directory not found: {self.input_dir}"
            )
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"[ERROR] Checkpoint directory not found: {self.checkpoint_dir}"
            )

    def run(self) -> None:
        """
        Executes the batch inference and saves results.
        """
        print("[INFO] Running monocular depth inference...")
        self.inferencer.run()
        print("[✓] Depth inference complete.")


def main() -> None:
    scene_dir = Path("lab_scene_05")

    dataset_path = Path(f"datasets/{scene_dir}")
    output_path = Path(f"results/{scene_dir}")
    input_dir = dataset_path / "rgb"
    checkpoint_dir = Path("checkpoints")
    encoder = "vits"

    app = DepthOfflineInference(
        input_dir=input_dir,
        output_dir=output_path,
        checkpoint_dir=checkpoint_dir,
        encoder=encoder
    )
    app.run()


if __name__ == "__main__":
    main()

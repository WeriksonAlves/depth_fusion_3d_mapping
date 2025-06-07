import json
from pathlib import Path
from typing import List, Dict

from frame_icp_aligner import FrameICPAligner


class FrameICPAlignerBatch:
    """
    Performs batch ICP alignment between monocular and real depth clouds
    for multiple frames and aggregates alignment metrics into a report.
    """

    def __init__(
        self,
        scene_name: str,
        frame_indices: List[int],
        voxel_size: float = 0.02,
        depth_scale: float = 1000.0
    ) -> None:
        self._scene_name = scene_name
        self._frame_indices = frame_indices
        self._voxel_size = voxel_size
        self._depth_scale = depth_scale

        self._dataset_dir = Path(f"datasets/{scene_name}")
        self._results_dir = Path(f"results/{scene_name}")
        self._metrics: Dict[int, Dict] = {}

    def _load_frame_metrics(self, index: int) -> Dict:
        """
        Loads ICP metrics for a specific frame from disk.

        Args:
            index (int): Frame index.

        Returns:
            Dict: Loaded metrics or status message if missing.
        """
        path = self._results_dir / "d6" / "icp_metrics.json"
        if not path.exists():
            return {"status": "missing"}

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_batch_report(self) -> None:
        """
        Saves a summary report of all ICP results.
        """
        output_dir = self._results_dir / "d6"
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / "icp_batch_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self._metrics, f, indent=4)

        print(f"\n[✓] Batch ICP report saved to: {report_path}")

    def run(self) -> None:
        """
        Executes batch ICP alignment for the given set of frame indices.
        """
        print(f"[INFO] Starting batch ICP alignment "
              f"on {len(self._frame_indices)} frames...")

        for index in self._frame_indices:
            print(f"\n[Frame {index:04d}]")
            try:
                aligner = FrameICPAligner(
                    dataset_dir=self._dataset_dir,
                    results_dir=self._results_dir,
                    frame_index=index,
                    voxel_size=self._voxel_size,
                    depth_scale=self._depth_scale
                )
                aligner.run()

                self._metrics[index] = self._load_frame_metrics(index)
            except Exception as e:
                print(f"[ERROR] Failed to process frame {index:04d}: {e}")
                self._metrics[index] = {"status": "error", "message": str(e)}

        self._save_batch_report()

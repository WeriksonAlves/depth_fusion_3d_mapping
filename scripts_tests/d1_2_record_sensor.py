import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from modules.utils.s0_realsense_recorder import RealSenseRecorder


def main() -> None:
    recorder = RealSenseRecorder(
        output_path="datasets/lab_scene_d",
        max_frames=60,
        fps=15
    )
    recorder.start()
    recorder.capture()


if __name__ == "__main__":
    main()

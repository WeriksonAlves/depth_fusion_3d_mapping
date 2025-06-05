import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from modules.utils.realsense_recorder import RealSenseRecorder


def main() -> None:
    recorder = RealSenseRecorder(
        output_dir="datasets/lab_scene_04",
        max_frames=60,
        fps=30
    )
    recorder.start()
    recorder.capture()


if __name__ == "__main__":
    main()

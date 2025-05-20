"""

"""

import os
import cv2

from modules.depth_estimator import DepthEstimator
from modules.point_cloud_processor import PointCloudProcessor
from modules.live_point_cloud_visualizer import LivePointCloudVisualizer


class DepthPointCloudApp:
    def __init__(self, mode='images', image_dir=None, output_dir=None, save_output=False, camera_idx=0):
        self.mode = mode
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.save_output = save_output
        self.camera_idx = camera_idx

        self.estimator = DepthEstimator()
        self.processor = PointCloudProcessor()
        self.visualizer = LivePointCloudVisualizer()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def process_frame(self, frame, name=None):
        depth = self.estimator.infer_depth(frame)
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
        vis_image = cv2.hconcat([cv2.flip(frame, 0), cv2.flip(depth_colored, 0)])

        if self.save_output and name:
            out_path = os.path.join(self.output_dir, f'depth_{name}')
            cv2.imwrite(out_path, vis_image)

        cv2.imshow('RGB + Depth Map', vis_image)

        pcd = self.processor.create_point_cloud(frame, depth)
        pcd_filtered = self.processor.filter_point_cloud(pcd)
        self.visualizer.update(pcd_filtered)

    def run(self):
        if self.mode == 'images':
            files = sorted(f for f in os.listdir(self.image_dir) if f.endswith('.png'))
            for filename in files:
                img_path = os.path.join(self.image_dir, filename)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"[Erro] Imagem corrompida: {img_path}")
                    continue
                self.process_frame(image, filename)
                if cv2.waitKey(1) == 27:
                    break

        elif self.mode == 'webcam':
            cap = cv2.VideoCapture(self.camera_idx)
            if not cap.isOpened():
                print("[Erro] Câmera não encontrada.")
                return
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[Aviso] Frame não capturado.")
                    break
                self.process_frame(frame, f'camera_frame_{frame_count:04d}.png')
                frame_count += 1
                if cv2.waitKey(1) == 27:
                    break
            cap.release()

        cv2.destroyAllWindows()
        self.visualizer.close()


if __name__ == '__main__':
    app = DepthPointCloudApp(
        mode='images',
        image_dir='SIBGRAPI2025/datasets/rgbd_dataset_freiburg1_xyz/rgb',
        output_dir='SIBGRAPI2025/results/depth_maps',
        save_output=False,
        camera_idx=1
    )
    app.run()

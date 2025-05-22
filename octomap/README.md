# Documentação de Integração com OctoMap no ROS 2 Humble

Este documento descreve as etapas de instalação, configuração e execução da integração entre nuvens de pontos geradas com Open3D e o mapeamento 3D usando `octomap_server` no ROS 2 Humble (Ubuntu 22.04). Inclui também correções aplicadas durante o processo de desenvolvimento.

---

## ✨ Visão Geral

O objetivo é gerar mapas de ocupação 3D a partir de imagens RGB utilizando um modelo de estimação de profundidade (DepthAnythingV2), gerar nuvens de pontos 3D com Open3D, e publicar essas nuvens em um tópico ROS para consumo pelo `octomap_server`, que constrói o Octomap.

---

## 🚀 Etapas Executadas

### 1. Criação do Workspace ROS 2

```bash
mkdir -p ~/octomap_ws/src
cd ~/octomap_ws
colcon build
source install/setup.bash
```

Adicionado ao `.bashrc`:

```bash
echo "source ~/octomap_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

### 2. Instalação do OctoMap

```bash
sudo apt update
sudo apt install ros-humble-octomap ros-humble-octomap-msgs ros-humble-octomap-server
```

---

### 3. Criação do pacote Python `o3d_publisher`

```bash
cd ~/octomap_ws/src
ros2 pkg create --build-type ament_python o3d_publisher --dependencies rclpy sensor_msgs std_msgs
```

Edição do `~/octomap_ws/src/o3d_publisher/setup.py`:

```python
    data_files=[
    	('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    	('share/' + package_name, ['package.xml']),
    	('share/' + package_name + '/launch', ['launch/octomap_launch.py']),
    ],
```

```python
    entry_points={
        'console_scripts': [
            'o3d_pub_node = o3d_publisher.o3d_pub_node:main',
        ],
    },
```

---

### 4. Criação do arquivo `o3d_pub_node.py`

O script `o3d_pub_node.py` é responsável por:

* Carregar nuvem de pontos `.ply` com Open3D
* Converter para `sensor_msgs/msg/PointCloud2`
* Publicar no tópico `o3d_points`

#### 🔧 Instruções de criação do script:

1. Crie o arquivo dentro da pasta do pacote:

```bash
nano ~/octomap_ws/src/o3d_publisher/o3d_publisher/o3d_pub_node.py
```

2. Cole o seguinte código completo no arquivo:

```python
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

import numpy as np
import struct

import open3d as o3d


def convert_cloud_to_ros_msg(cloud: o3d.geometry.PointCloud, stamp, frame_id="map") -> PointCloud2:
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)

    if colors.shape[0] != points.shape[0]:
        colors = np.zeros_like(points)

    data = []
    for i in range(points.shape[0]):
        x, y, z = points[i]
        r, g, b = (colors[i] * 255).astype(np.uint8)
        rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, 0))[0]
        data.append([x, y, z, rgb])
    data = np.array(data, dtype=np.float32)

    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points.shape[0]
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * points.shape[0]
    msg.is_dense = True
    msg.data = data.tobytes()
    return msg


class O3DPublisher(Node):
    def __init__(self):
        super().__init__('o3d_publisher')
        self.publisher_ = self.create_publisher(PointCloud2, 'o3d_points', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)  # 1 Hz

        self.get_logger().info('Open3D publisher ready.')

    def timer_callback(self):
        try:
            pcd = o3d.io.read_point_cloud("/octomap_ws/points/pcd.ply") # Mude para o caminho dos seus dados
            stamp = self.get_clock().now().to_msg()
            msg = convert_cloud_to_ros_msg(pcd, stamp)
            self.publisher_.publish(msg)
            self.get_logger().info('Nuvem publicada!')
        except Exception as e:
            self.get_logger().error(f"Erro ao publicar nuvem: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = O3DPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

3. Salve e feche o arquivo (Ctrl+O, Enter, Ctrl+X).

---

### 5. Arquivo de Launch para o Octomap

1. Crie o arquivo `launch/octomap_launch.py`:

```bash
mkdir -p ~/octomap_ws/src/o3d_publisher/launch
nano ~/octomap_ws/src/o3d_publisher/launch/octomap_launch.py
```

2. Cole o seguinte código completo no arquivo:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='octomap_server',
            executable='octomap_server_node',
            name='octomap_server',
            output='screen',
            parameters=[{
                'frame_id': 'map',
                'resolution': 0.05,
                'sensor_model_max_range': 5.0
            }],
            remappings=[
                ('cloud_in', 'o3d_points')
            ]
        )
    ])
```

3. Recompile o workspace:

```bash
cd ~/octomap_ws
colcon build --packages-select o3d_publisher
source install/setup.bash
```

---

### 6. Correções durante o desenvolvimento (Verificar durante a instalção)

* Erro de falta do pacote `open3d`:

  * Solução: `pip3 install open3d`

* Erro de incompatibilidade com `numpy 2.x`:

  * Solução: fazer downgrade:

```bash
pip3 install "numpy<2" --force-reinstall
```

---



### 7. Execução

#### Publicador da nuvem:

```bash
ros2 run o3d_publisher o3d_pub_node
```

#### Servidor do Octomap:

```bash
ros2 launch o3d_publisher octomap_launch.py
```

#### Visualização no RViz:

```bash
rviz2
```
* Fix Frame: `map`
* Add: `/occupied_cells_vis_array` (tipo: `MarkerArray`)

---

## 🎓 Resultado

Com os tópicos corretamente conectados e os dados da nuvem em formato esperado, o `octomap_server` constrói dinamicamente o mapa 3D do ambiente, que pode ser visualizado em tempo real no RViz.

---

## 📍 Considerações finais

Esse setup serve como base para uma pipeline completa de reconstrução de ambientes 3D com uma câmera RGB. Etapas futuras podem incluir integração com VIO/SLAM, movimentação do robô e exportação do mapa final.

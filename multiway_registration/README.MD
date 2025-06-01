# 📄 **Documentação das Etapas Realizadas — Dias 1 a 3**

### 🗓️ Contexto: Cronograma de 2 Semanas para Reconstrução Multi-Sensorial

Objetivo geral: comparar reconstruções 3D a partir de sensores RGB-D e profundidade monocular usando o algoritmo de **multiway registration da Open3D**.

---

## ✅ **🔹 Dia 1 — Planejamento e Aquisição de Dados**

### 🎯 Objetivo:

Realizar a aquisição de dados RGB-D utilizando a câmera Intel RealSense D435 com ROS 2 e salvar os dados em formato `.bag`.

### 🔧 Procedimentos esperados:

* Conectar a D435 ao notebook.
* Lançar o driver da câmera com parâmetros desejados:

  ```bash
  ros2 launch realsense2_camera rs_camera.launch.py \
      enable_depth:=true \
      enable_color:=true \
      align_depth.enable:=true \
      depth_module.profile:=640x480x10 \
      color_module.profile:=640x480x10
  ```
* Gravar dados com:

  ```bash
  ros2 bag record /camera/color/image_raw /camera/aligned_depth_to_color/image_raw /camera/color/camera_info -o datasets/lab_scene_01
  ```

### ⚠️ Problema enfrentado:

> Após a execução, percebeu-se que os tópicos da câmera **não estavam ativos** e o `.bag` gravado estava **incompleto ou vazio**.

### ✅ Solução alternativa:

> Para não atrasar o cronograma, adotou-se um dataset do Kinect (TUM RGB-D `freiburg1_xyz`) como substituto temporário.

---

## ✅ **🔹 Dia 2 — Conversão e Organização dos Dados**

### 🎯 Objetivo:

Extrair imagens RGB, mapas de profundidade e parâmetros de câmera do `.bag`, estruturando-os para o pipeline de reconstrução.

### 🔧 Procedimentos esperados com `.bag` da D435:

* Usar `rosbags` para ler `.db3` e exportar:

  * RGB → `.png`
  * Depth → `.png` (16-bit)
  * Camera info → `intrinsics.json`

### ⚠️ Problema enfrentado:

> O `.bag` do Kinect estava no formato **ROS 1 (não ROS 2)** e não era compatível com `ros2 bag`.

### ✅ Solução alternativa:

> Como o dataset TUM já fornece arquivos separados, a extração foi desnecessária. O dataset foi estruturado como:

```
datasets/lab_scene_kinect/
├── rgb/
├── depth/
├── rgb.txt
├── depth.txt
├── groundtruth.txt
└── intrinsics.json   ✅ criado manualmente
```

### 📄 `intrinsics.json` (valores padrão para Kinect da TUM):

```json
{
  "width": 640,
  "height": 480,
  "K": [517.3, 0.0, 318.6,
        0.0, 516.5, 255.3,
        0.0, 0.0, 1.0],
  "D": [0, 0, 0, 0, 0],
  "distortion_model": "plumb_bob"
}
```

---

## ✅ **🔹 Dia 3 — Reconstrução com Profundidade Real**

### 🎯 Objetivo:

Aplicar o pipeline de **multiway registration** da Open3D utilizando as profundidades reais do Kinect (substituindo temporariamente a D435).

### 🧠 Etapas implementadas:

1. Carregamento de RGB + profundidade.
2. Criação da nuvem de pontos por quadro com intrínsecos corretos.
3. Downsample + cálculo de normais + FPFH (implícito).
4. Registro par-a-par (ICP).
5. Otimização global do grafo de poses.
6. Mesclagem das nuvens transformadas.
7. Salvamento como `reconstruction_d435.ply`.

### 🐍 Script utilizado:

> `multiway_registration_real_depth.py`

📁 Salva a reconstrução em:

```bash
datasets/lab_scene_kinect/reconstruction_d435.ply
```

📈 Também imprime:

* Número total de pontos.
* Volume da caixa delimitadora (AABB).

---

## ✅ Visualização da reconstrução

### 🧭 Escolha do modo:

* `--mode open3d`: janela interativa local.
* `--mode rviz`: publicação como `/reconstruction_points` (ROS 2).

### 🐍 Script utilizado:

> `reconstruction_viewer.py`

🔧 Dependência ROS 2: `rclpy`, `sensor_msgs`, `std_msgs`.

### Exemplo de execução:

```bash
python reconstruction_viewer.py --mode open3d
python reconstruction_viewer.py --mode rviz
```

---

## ✅ Conclusão (Parcial)

Mesmo com a falha de aquisição inicial com a RealSense D435, foi possível:

* Simular todo o pipeline com um dataset RGB-D compatível.
* Validar o processo de reconstrução com profundidade real.
* Obter reconstrução global funcional.

---

## 📝 Observações Finais:

* Quando for possível repetir a aquisição com a D435, basta:

  * Substituir os diretórios `rgb/`, `depth/`, e `intrinsics.json`.
  * Reexecutar o script `multiway_registration_real_depth.py`.

---
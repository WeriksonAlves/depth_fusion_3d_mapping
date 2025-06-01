## 🗓️ Roteiro de Execução do Trabalho (14 dias)

### 🔹 **Semana 1 — Coleta, Pré-processamento e Reconstruções Iniciais**

#### 📅 **Dia 1 — Planejamento e Aquisição**

* Definir a trajetória ou ambiente de coleta (ex.: sala, corredor).
* Configurar a **Intel RealSense D435** com `realsense2_ros` ou `pyrealsense2`.
* Coletar imagens RGB e profundidade (formato `.bag`, `.png` + `.npy`, ou `.ply`).
* Salvar dados com timestamps sincronizados.

#### 📅 **Dia 2 — Conversão e Organização dos Dados**

* Extrair frames RGB e mapas de profundidade reais da D435.
* Garantir estrutura de pastas: `RGB/`, `Depth_D435/`, `Intrinsics.json`.
* Verificar e salvar os intrínsecos reais da D435.

#### 📅 **Dia 3 — Reconstrução com D435 (Depth Real)**

* Aplicar o pipeline de **Multiway Registration** com os dados da D435:

  * Downsample + FPFH.
  * Pairwise Registration + ICP.
  * Grafo de poses.
  * Reconstrução final.
* Salvar nuvem como `reconstruction_d435.ply`.
* Documentar qualidade da reconstrução (visual + métrica: nº pontos, volume, etc).

#### 📅 **Dia 4 — Inferência Monocular (DepthAnythingV2)**

* Rodar `DepthAnythingV2` nos frames RGB.
* Salvar profundidades estimadas como `.npy` ou `.png`.
* Verificar escala inconsistente com D435 (esperado).

#### 📅 **Dia 5 — Reconstrução com DepthAnythingV2**

* Aplicar Multiway Registration com as profundidades estimadas:

  * Ajustar intrínsecos se necessário.
  * Usar mesmo pipeline da D435.
* Salvar como `reconstruction_depthanything.ply`.
* Documentar distorções, falhas de escala e resultados visuais.

#### 📅 **Dia 6 — Estudo de Escala e Alinhamento**

* Calcular a diferença de escala entre as profundidades.
* Tentar alinhar uma nuvem da DepthAnything com uma da D435 usando ICP (Open3D).
* Registrar a transformação `T_d_to_m` (depth monocular para métrica).

#### 📅 **Dia 7 — Revisão Parcial**

* Consolidar reconstruções `D435` vs `DepthAnythingV2`.
* Avaliar se dados estão consistentes e salvá-los para reprodutibilidade.
* Ajustar pipeline se necessário.

---

### 🔹 **Semana 2 — Fusão, Comparação e Escrita**

#### 📅 **Dia 8 — Fusão de Profundidades**

* Projetar método de fusão:

  * Warpar a profundidade monocular para o frame da D435.
  * Combinar via média ponderada ou substituição condicional (ex: profundidade válida e menor).
* Salvar nova profundidade como `fused_depth.png`.

#### 📅 **Dia 9 — Reconstrução com Profundidade Fundida**

* Aplicar Multiway Registration com a profundidade fundida.
* Salvar como `reconstruction_fused.ply`.
* Comparar visualmente e numericamente com as anteriores.

#### 📅 **Dia 10 — Métricas de Avaliação**

* Métricas sugeridas:

  * Número de pontos finais.
  * Densidade média da nuvem.
  * Volume do AABB.
  * Alinhamento com ground-truth (se existir).
* Anotar comparações qualitativas (ruído, buracos, descontinuidades).

#### 📅 **Dia 11 — Preparação de Gráficos e Visualizações**

* Visualizar todas as reconstruções em RViz ou Open3D.
* Criar colagens de comparação lado a lado.
* Plotar curvas de volume vs. técnica.

#### 📅 **Dia 12 — Escrita do Relatório**

* Introdução, motivação, sensores utilizados.
* Descrição do pipeline de Multiway Registration.
* Comparações: D435, Monocular, Fusão.

#### 📅 **Dia 13 — Conclusões e Discussões**

* Discutir limitações do monocular (escala, ruído).
* Vantagens e problemas da fusão.
* Sugestões para melhorias futuras (ex: aprendizado da escala, refinamento geométrico).

#### 📅 **Dia 14 — Revisão e Backup**

* Revisar todo o código e relatório.
* Criar backup do projeto e dados.
* Preparar versão final para submissão/apresentação.


---

## ✅ **Checklist de Revisão Parcial**

### 1. **Reconstruções obtidas até agora**

| Tipo                     | Arquivo                                    | Observação                          |
| ------------------------ | ------------------------------------------ | ----------------------------------- |
| D435 (real)              | `reconstruction_d435.ply`                  | Escala métrica correta              |
| DepthAnything (bruta)    | `reconstruction_depthanything.ply`         | Sem escala absoluta                 |
| DepthAnything (alinhada) | `reconstruction_depthanything_aligned.ply` | Após ICP + escala ajustada          |
| Transformação ICP        | `T_d_to_m.npy`                             | Matriz de transformação salva (4x4) |

---

### 2. **Critérios de Avaliação da Reconstrução**

Use os seguintes critérios qualitativos e quantitativos:

| Critério                   | Como avaliar                                        |
| -------------------------- | --------------------------------------------------- |
| **Número de pontos**       | `len(pcd.points)`                                   |
| **Volume da reconstrução** | `pcd.get_axis_aligned_bounding_box().volume()`      |
| **Sobreposição visual**    | `compare_reconstructions.py` e visualizações        |
| **Preservação geométrica** | Observe superfícies planas, objetos bem delimitados |
| **Distorções monocular**   | Curvaturas, descontinuidades, ausência de paralaxe  |

---

### 3. **Verificação de consistência de arquivos**

Certifique-se de que os seguintes arquivos existem e estão completos:

```bash
datasets/lab_scene_kinect/
├── rgb/
├── depth_d435/
├── depth_mono/
├── intrinsics.json
├── reconstruction_d435.ply
├── reconstruction_depthanything.ply
├── reconstruction_depthanything_aligned.ply
├── T_d_to_m.npy
```

---

### 4. **Validação dos scripts do pipeline**

| Etapa                    | Script usado                          |
| ------------------------ | ------------------------------------- |
| Inferência monocular     | `infer_depth_anything.py`             |
| Reconstrução Depth Real  | `multiway_registration_real_depth.py` |
| Reconstrução Monocular   | `multiway_registration_mono_depth.py` |
| Visualização comparativa | `compare_reconstructions.py`          |
| Alinhamento via ICP      | `align_depth_mono_to_real.py`         |

---

## ✏️ Sugestão de ajustes no pipeline

| Item                | Ajuste sugerido                                      | Quando aplicar?            |
| ------------------- | ---------------------------------------------------- | -------------------------- |
| `depth_trunc`       | Reduzir de 4.0m → 2.5m para indoor com DepthAnything | Se muitas regiões esparsas |
| `voxel_size`        | Reduzir para 0.01m para mais detalhes                | Após validação visual      |
| `scale mono` manual | Usar fator aproximado antes do ICP                   | Para visualizações rápidas |

---

## 🗂️ Reprodutibilidade

Recomenda-se salvar também:

* `requirements.txt` atualizado.
* Versão do modelo DepthAnythingV2 utilizada (`vits`).
* Scripts principais com hash ou tags.

Exemplo de comando para salvar dependências:

```bash
pip freeze > results/env_requirements.txt
```

---

## ✅ Deseja seguir agora para o **📅 Dia 8 — Fusão de Profundidades**?

Posso gerar um script para:

1. Carregar as profundidades da D435 e DepthAnythingV2.
2. Aplicar `T_d_to_m` na monocular.
3. Realizar uma fusão ponto a ponto (por confiança, média ou substituição).
4. Salvar novo conjunto em `depth_fused/`.

Com base no algoritmo de **Multiway Registration** da Open3D e nos objetivos experimentais, segue um roteiro de execução detalhado com **prazo máximo de 2 semanas**, dividido em **etapas diárias**, priorizando clareza, organização e foco experimental.

---

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



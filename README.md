# SIBGRAPI2025

\subsection{Nuvem de Pontos 3D}

A partir da imagem RGB e do mapa de profundidade inferido pelo modelo, aplicamos a técnica de \textit{back-projection} para gerar uma nuvem de pontos 3D. Utilizamos a biblioteca Open3D, empregando uma matriz intrínseca do tipo pinhole para transformar as coordenadas de pixel em coordenadas espaciais reais.

O mapa de profundidade foi convertido para um formato compatível com a Open3D, usando escala métrica e truncamento de profundidade para remover valores extremos. Em seguida, aplicamos filtros de downsampling voxelizado e remoção estatística de outliers para melhorar a densidade e a consistência da nuvem gerada.

A Figura~\ref{fig:pointcloud} apresenta a visualização da nuvem resultante no ambiente real capturado.

# Graph Neural Networks for Structure-from-Motion via Low-Rank Matrix Factorization

Structure-from-Motion (SfM) reconstructs 3D structure and camera poses from 2D keypoints across multiple views. Classical pipelines rely on a sequence of geometric subproblems and bundle adjustment. Recent work has shown that graph neural networks (GNNs) can directly learn SfM-specific primitives, providing competitive accuracy at lower runtime. In parallel, graph-based formulations of low-rank matrix factorization have demonstrated that GNNs can accelerate matrix computations by leveraging bipartite graph representations. 

This project will explore the intersection of these two ideas by formulating SfM as a low-rank matrix factorization problem and applying graph neural networks to jointly estimate 3D structure and camera parameters. The goal is to design and evaluate a GNN architecture that combines SfM-specific graph attention with bipartite self-attention layers and optimization-inspired updates.

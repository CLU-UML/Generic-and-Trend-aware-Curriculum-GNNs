# Generic and Trend-aware Curriculum Learning for Relation Extraction

GTNN is a generic and trend-aware curriculum learning approach that effectively integrates textual and structural information in text graphs for relation extraction between entities, which we consider as node pairs in graphs. The proposed model extends existing curriculum learning approaches by incorporating sample-level loss trends to better discriminate easier from harder samples and schedule them for training.

GTNN has three main componenets 1) module to learn from the graph structure, 2) an operator to add additional features, 3) a curriculum to dynamically decide easy and hard examples while training the model.

![alt text](https://github.com/CLU-UML/gtnn/blob/main/architecture_diagram_gtnn_trend.png)

The architecture of the proposed graph text neural network (GTNN) model with Trend-SL curriculum learning approach. The proposed model consists of an encoder-decoder component that determines relations between given node pairs. The graph neural encoder takes as input features from textual descriptions of nodes and sub-graph extracted for a given node pair to create node embeddings. The resulting embeddings in conjunction with additional text features are {\em directly} used by the decoder to predict links between given entity pairs. The resulting loss is given as an input to our Trend-SL approach to dynamically learn a curriculum during training.




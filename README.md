# Generic and Trend-aware Curriculum Learning for Relation Extraction

Graph Text Neural Network (GTNN) is a generic and trend-aware curriculum learning approach that effectively integrates textual and structural information in text graphs for relation extraction between entities, which we consider as node pairs in graphs. The proposed model extends existing curriculum learning approaches by incorporating sample-level loss trends to better discriminate easier from harder samples and schedule them for training.

GTNN has three main componenets 1) Graph Neural Encoder: module to learn from the graph structure, 2) Textual Information Encoder: module to add additional features, 3) Curriculum based Graph Neural Decoder: a curriculum to dynamically decide easy and hard examples while training the model.

<p align="center">
<img src="https://github.com/CLU-UML/gtnn/blob/main/architecture_diagram_gtnn_trend.png" width="900" height="600">
</p>

The architecture of the proposed graph text neural network (GTNN) model with Trend-SL curriculum learning approach. The proposed model consists of an encoder-decoder component that determines relations between given node pairs. The graph neural encoder takes as input features from textual descriptions of nodes and sub-graph extracted for a given node pair to create node embeddings. The resulting embeddings in conjunction with additional text features are directly used by the decoder to predict links between given entity pairs. The resulting loss is given as an input to our Trend-SL approach to dynamically learn a curriculum during training.

# Data

There are two datasets: PGR and GDPR. 

* **Phenotype Gene Relation (PGR):**  PGR is created by Sousa et al., NAACL 2019 (https://aclanthology.org/N19-1152/) from PubMed articles and contains sentences describing relations between given genes and phenotypes. In our experiments, we only include data samples in PGR with available text descriptions for their genes and phenotypes. This amounts to ~71% of the original dataset. 

* **Gene, Disease, Phenotype Relation (GDPR):** This dataset is obtained by combining and linking entities across two freely-available datasets: Online Mendelian Inheritance in Man (OMIM, https://omim.org/) and Human Phenotype Ontology (HPO, https://hpo.jax.org/). The dataset contains relations between genes, diseases and phenotypes.

To download datasets with embeddings and Train/Test/Val splits, go to data directory and run download.sh as follows

```
sh ./download.sh
```

# To run the code
Use the following command with appropriate arguments:

```
python main.py --dataset=pgr
```

# Citation
```
@inproceedings{nidhi-etal-2022-gtnn,
    title = "Generic and Trend-aware Curriculum Learning for Relation Extraction",
    author = "Vakil, Nidhi and  Amiri, Hadi",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL)",
    publisher = "Association for Computational Linguistics",
    year = "2022"
    
}
```

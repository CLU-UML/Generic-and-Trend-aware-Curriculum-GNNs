#!/bin/bash
pip install gdown

echo 'Downloading PGR Train/Test/Val split data with embeddings (6GB)' 
mkdir "pgr"
wget https://clu.cs.uml.edu/data/pgr_data/pgr_train_test_val_doc2vec.pkl
mv pgr_train_test_val_doc2vec.pkl ./pgr/

echo 'Downloading GDPR Train/Test/Val split data with embeddings (3GB)'
mkdir gdpr
wget https://clu.cs.uml.edu/data/omim_data/gdpr_train_test_val_doc2vec.pkl
mv gdpr_train_test_val_doc2vec.pkl ./gdpr/

echo 'Downloading PGR Train/Test/Val split data with embeddings (6GB)' 
mkdir pgr
wget http://linktopgr.onedrive.com
mv pgr_train_test_val_doc2vec.pkl ./pgr/


echo 'Downloading GDPR Train/Test/Val split data with embeddings (3GB)'
mkdir gdpr
wget http://linktopgr.onedrive.com
mv gdpr_train_test_val_doc2vec.pkl ./gdpr/

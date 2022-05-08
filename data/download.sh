echo 'Downloading PGR Train/Test/Val split data with embeddings (6GB)' 
mkdir pgr
wget https://studentuml-my.sharepoint.com/:u:/g/personal/nidhipiyush_vakil_student_uml_edu/Eejm6ZaqMDhIluR8KeoM7ZgBTPiddibQBubyaXpUAuHvPQ?e=Yck1ML
mv pgr_train_test_val_doc2vec.pkl ./pgr/


echo 'Downloading GDPR Train/Test/Val split data with embeddings (3GB)'
mkdir gdpr
wget https://studentuml-my.sharepoint.com/:u:/g/personal/nidhipiyush_vakil_student_uml_edu/EfUq4dcTfZhFjPc_KwqRLlQBY2KC4ci4H2uLily-cSXGRQ?e=DajweP
mv gdpr_train_test_val_doc2vec.pkl ./gdpr/

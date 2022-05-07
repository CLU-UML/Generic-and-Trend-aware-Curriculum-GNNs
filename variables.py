import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dir_data = "/GTNN/data"
dir_model = "/GTNN/saved_models"
dir_logs = "/GTNN/logs"

import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dir_data = "/gtnn/data"
dir_model = "/gtnn/saved_models"
dir_logs = "/gtnn/logs"

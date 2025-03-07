# 帮助函数
import torch

def get_device(cuda_device=0):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cuda_device}")
    else:
        device = torch.device("cpu")
    return device

class ResultWriter:
    def __init__(self, output_file):
        self.output_file = output_file

    def log(self, message):
        with open(self.output_file, 'a') as f:
            f.write(message + "\n")

def log(message):
    print(message)
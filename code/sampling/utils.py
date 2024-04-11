from datetime import datetime
import torch
import gc

ts = lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log = lambda x: print(f'[{ts()}] {x}')

def clean_memory(device):
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif str(device) == 'mps':
        torch.mps.empty_cache()
    else:
        pass
    gc.collect()

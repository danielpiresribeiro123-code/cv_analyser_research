import torch
print(torch.cuda.is_available())
print(torch.version.cuda) # type: ignore
print(torch.cuda.get_device_name(0))

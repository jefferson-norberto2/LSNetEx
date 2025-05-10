from torch import device, load
from torch.cuda import is_available

from lsnetex.models.LSNetEx import LSNetEx
from lsnetex.config import opt


model_path = opt.model_path

my_device = device("cuda" if is_available() else "cpu")
print('Device in use:', my_device)

# Carrega o modelo
model = LSNetEx(network=opt.network).to(my_device)
model.load_state_dict(load(model_path, map_location=my_device))
model.eval()

# ver a quantidade de parâmetros do modelo
total_params = sum(p.numel() for p in model.parameters())
print(f"Quantidade total de parâmetros do modelo: {total_params}")


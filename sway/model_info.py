import torch
# Enter the EXACT path to your model (the weight file).
MODELPATH='weights/taylorswiftdebut_model.pth'
model_params = torch.load(MODELPATH)
param_names = list(model_params.keys())
print(param_names)
print("======")
for key, value in model_params.items():
    if key == 'info':
        print('>>>>> Epochs: ', value)
    elif key == 'sr':
        print('>>>>> Sample Rate: ', value)
    else:
        print(key, str(value)[:1000])
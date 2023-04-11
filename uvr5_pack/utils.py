import torch
import numpy as np
from tqdm import tqdm

def make_padding(width, cropsize, offset):
    left = offset
    roi_size = cropsize - left * 2
    if roi_size == 0:
        roi_size = cropsize
    right = roi_size - (width % roi_size) + left

    return left, right, roi_size
def inference(X_spec, device, model, aggressiveness,data):
    '''
    data ： dic configs
    '''
    
    def _execute(X_mag_pad, roi_size, n_window, device, model, aggressiveness,is_half=True):
        model.eval()
        with torch.no_grad():
            preds = []
            
            iterations = [n_window]

            total_iterations = sum(iterations)            
            for i in tqdm(range(n_window)): 
                start = i * roi_size
                X_mag_window = X_mag_pad[None, :, :, start:start + data['window_size']]
                X_mag_window = torch.from_numpy(X_mag_window)
                if(is_half):X_mag_window=X_mag_window.half()
                X_mag_window=X_mag_window.to(device)

                pred = model.predict(X_mag_window, aggressiveness)

                pred = pred.detach().cpu().numpy()
                preds.append(pred[0])
                
            pred = np.concatenate(preds, axis=2)
        return pred
    
    def preprocess(X_spec):
        X_mag = np.abs(X_spec)
        X_phase = np.angle(X_spec)

        return X_mag, X_phase
    
    X_mag, X_phase = preprocess(X_spec)

    coef = X_mag.max()
    X_mag_pre = X_mag / coef

    n_frame = X_mag_pre.shape[2]
    pad_l, pad_r, roi_size = make_padding(n_frame,
                                                data['window_size'], model.offset)
    n_window = int(np.ceil(n_frame / roi_size))

    X_mag_pad = np.pad(
        X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')

    if(list(model.state_dict().values())[0].dtype==torch.float16):is_half=True
    else:is_half=False
    pred = _execute(X_mag_pad, roi_size, n_window,
                        device, model, aggressiveness,is_half)
    pred = pred[:, :, :n_frame]
    
    if data['tta']:
        pad_l += roi_size // 2
        pad_r += roi_size // 2
        n_window += 1

        X_mag_pad = np.pad(
            X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')

        pred_tta = _execute(X_mag_pad, roi_size, n_window,
                                device, model, aggressiveness,is_half)
        pred_tta = pred_tta[:, :, roi_size // 2:]
        pred_tta = pred_tta[:, :, :n_frame]

        return (pred + pred_tta) * 0.5 * coef, X_mag, np.exp(1.j * X_phase)
    else:
        return pred * coef, X_mag, np.exp(1.j * X_phase)
            


def  _get_name_params(model_path , model_hash):
    ModelName = model_path
    if model_hash == '47939caf0cfe52a0e81442b85b971dfd':  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/4band_44100.json')
        param_name_auto=str('4band_44100')
    if model_hash == '4e4ecb9764c50a8c414fee6e10395bbe':  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/4band_v2.json')
        param_name_auto=str('4band_v2')
    if model_hash == 'ca106edd563e034bde0bdec4bb7a4b36':
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/4band_v2.json')
        param_name_auto=str('4band_v2')
    if model_hash == 'e60a1e84803ce4efc0a6551206cc4b71':
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/4band_44100.json')
        param_name_auto=str('4band_44100')
    if model_hash == 'a82f14e75892e55e994376edbf0c8435':  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/4band_44100.json')
        param_name_auto=str('4band_44100')
    if model_hash == '6dd9eaa6f0420af9f1d403aaafa4cc06':   
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/4band_v2_sn.json')
        param_name_auto=str('4band_v2_sn')
    if model_hash == '08611fb99bd59eaa79ad27c58d137727':
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/4band_v2_sn.json')
        param_name_auto=str('4band_v2_sn')
    if model_hash == '5c7bbca45a187e81abbbd351606164e5':
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/3band_44100_msb2.json')
        param_name_auto=str('3band_44100_msb2')
    if model_hash == 'd6b2cb685a058a091e5e7098192d3233':    
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/3band_44100_msb2.json')
        param_name_auto=str('3band_44100_msb2')
    if model_hash == 'c1b9f38170a7c90e96f027992eb7c62b': 
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/4band_44100.json')
        param_name_auto=str('4band_44100')
    if model_hash == 'c3448ec923fa0edf3d03a19e633faa53':  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/4band_44100.json')
        param_name_auto=str('4band_44100')
    if model_hash == '68aa2c8093d0080704b200d140f59e54':  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/3band_44100.json')
        param_name_auto=str('3band_44100.json')
    if model_hash == 'fdc83be5b798e4bd29fe00fe6600e147':  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/3band_44100_mid.json')
        param_name_auto=str('3band_44100_mid.json')
    if model_hash == '2ce34bc92fd57f55db16b7a4def3d745':  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/3band_44100_mid.json')
        param_name_auto=str('3band_44100_mid.json')
    if model_hash == '52fdca89576f06cf4340b74a4730ee5f':  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/4band_44100.json')
        param_name_auto=str('4band_44100.json')
    if model_hash == '41191165b05d38fc77f072fa9e8e8a30':  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/4band_44100.json')
        param_name_auto=str('4band_44100.json')
    if model_hash == '89e83b511ad474592689e562d5b1f80e':  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/2band_32000.json')
        param_name_auto=str('2band_32000.json')
    if model_hash == '0b954da81d453b716b114d6d7c95177f':  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/2band_32000.json')
        param_name_auto=str('2band_32000.json')

    #v4 Models    
    if model_hash == '6a00461c51c2920fd68937d4609ed6c8':  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/1band_sr16000_hl512.json')
        param_name_auto=str('1band_sr16000_hl512')
    if model_hash == '0ab504864d20f1bd378fe9c81ef37140':  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/1band_sr32000_hl512.json')
        param_name_auto=str('1band_sr32000_hl512')
    if model_hash == '7dd21065bf91c10f7fccb57d7d83b07f':  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/1band_sr32000_hl512.json')
        param_name_auto=str('1band_sr32000_hl512')
    if model_hash == '80ab74d65e515caa3622728d2de07d23':  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/1band_sr32000_hl512.json')
        param_name_auto=str('1band_sr32000_hl512')
    if model_hash == 'edc115e7fc523245062200c00caa847f':  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/1band_sr33075_hl384.json')
        param_name_auto=str('1band_sr33075_hl384')
    if model_hash == '28063e9f6ab5b341c5f6d3c67f2045b7':  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/1band_sr33075_hl384.json')
        param_name_auto=str('1band_sr33075_hl384')
    if model_hash == 'b58090534c52cbc3e9b5104bad666ef2':  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/1band_sr44100_hl512.json')
        param_name_auto=str('1band_sr44100_hl512')
    if model_hash == '0cdab9947f1b0928705f518f3c78ea8f':  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/1band_sr44100_hl512.json')
        param_name_auto=str('1band_sr44100_hl512')
    if model_hash == 'ae702fed0238afb5346db8356fe25f13':  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/1band_sr44100_hl1024.json')
        param_name_auto=str('1band_sr44100_hl1024')                        
    #User Models

    #1 Band
    if '1band_sr16000_hl512' in ModelName:  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/1band_sr16000_hl512.json')
        param_name_auto=str('1band_sr16000_hl512')
    if '1band_sr32000_hl512' in ModelName:  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/1band_sr32000_hl512.json')
        param_name_auto=str('1band_sr32000_hl512')
    if '1band_sr33075_hl384' in ModelName:  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/1band_sr33075_hl384.json')
        param_name_auto=str('1band_sr33075_hl384')
    if '1band_sr44100_hl256' in ModelName:  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/1band_sr44100_hl256.json')
        param_name_auto=str('1band_sr44100_hl256')
    if '1band_sr44100_hl512' in ModelName:  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/1band_sr44100_hl512.json')
        param_name_auto=str('1band_sr44100_hl512')
    if '1band_sr44100_hl1024' in ModelName:  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/1band_sr44100_hl1024.json')
        param_name_auto=str('1band_sr44100_hl1024')
        
    #2 Band
    if '2band_44100_lofi' in ModelName:  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/2band_44100_lofi.json')
        param_name_auto=str('2band_44100_lofi')
    if '2band_32000' in ModelName:  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/2band_32000.json')
        param_name_auto=str('2band_32000')
    if '2band_48000' in ModelName:  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/2band_48000.json')
        param_name_auto=str('2band_48000')
        
    #3 Band   
    if '3band_44100' in ModelName:  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/3band_44100.json')
        param_name_auto=str('3band_44100')
    if '3band_44100_mid' in ModelName:  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/3band_44100_mid.json')
        param_name_auto=str('3band_44100_mid')
    if '3band_44100_msb2' in ModelName:  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/3band_44100_msb2.json')
        param_name_auto=str('3band_44100_msb2')
        
    #4 Band    
    if '4band_44100' in ModelName:  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/4band_44100.json')
        param_name_auto=str('4band_44100')
    if '4band_44100_mid' in ModelName:  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/4band_44100_mid.json')
        param_name_auto=str('4band_44100_mid')
    if '4band_44100_msb' in ModelName:  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/4band_44100_msb.json')
        param_name_auto=str('4band_44100_msb')
    if '4band_44100_msb2' in ModelName:  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/4band_44100_msb2.json')
        param_name_auto=str('4band_44100_msb2')
    if '4band_44100_reverse' in ModelName:  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/4band_44100_reverse.json')
        param_name_auto=str('4band_44100_reverse')
    if '4band_44100_sw' in ModelName:  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/4band_44100_sw.json') 
        param_name_auto=str('4band_44100_sw')
    if '4band_v2' in ModelName:  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/4band_v2.json')
        param_name_auto=str('4band_v2')
    if '4band_v2_sn' in ModelName:  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/4band_v2_sn.json')
        param_name_auto=str('4band_v2_sn')
    if 'tmodelparam' in ModelName:  
        model_params_auto=str('uvr5_pack/lib_v5/modelparams/tmodelparam.json')
        param_name_auto=str('User Model Param Set')
    return param_name_auto , model_params_auto

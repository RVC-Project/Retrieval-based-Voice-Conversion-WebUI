import PySimpleGUI as sg
import sounddevice as sd
import noisereduce as nr
import numpy as np
from fairseq import checkpoint_utils
import librosa,torch,parselmouth,faiss,time,threading
import torch.nn.functional as F
#import matplotlib.pyplot as plt
from infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono
from webui_locale import I18nAuto
i18n = I18nAuto()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RVC:
    def __init__(self,key,pth_path,index_path,npy_path) -> None:
        '''
        初始化
        '''
        self.f0_up_key=key
        self.time_step = 160 / 16000 * 1000
        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.index=faiss.read_index(index_path)
        self.big_npy=np.load(npy_path)
        model_path = "TEMP\\hubert_base.pt"
        print("load model(s) from {}".format(model_path))
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [model_path],
            suffix="",
        )
        self.model = models[0]
        self.model = self.model.to(device)
        self.model = self.model.half()
        self.model.eval()
        cpt = torch.load(pth_path, map_location="cpu")
        tgt_sr = cpt["config"][-1]
        cpt["config"][-3]=cpt["weight"]["emb_g.weight"].shape[0]#n_spk
        if_f0=cpt.get("f0",1)
        if(if_f0==1):
            self.net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=True)
        else:
            self.net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
        del self.net_g.enc_q
        print(self.net_g.load_state_dict(cpt["weight"], strict=False))
        self.net_g.eval().to(device)
        self.net_g.half()


    def get_f0_coarse(self,f0):
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        # f0_mel[f0_mel > 188] = 188
        f0_coarse = np.rint(f0_mel).astype(np.int)
        return f0_coarse
    
    def get_f0(self,x, p_len,f0_up_key=0):
        f0 = parselmouth.Sound(x, 16000).to_pitch_ac(
            time_step=self.time_step / 1000, voicing_threshold=0.6,
            pitch_floor=self.f0_min, pitch_ceiling=self.f0_max).selected_array['frequency']

        pad_size=(p_len - len(f0) + 1) // 2
        if(pad_size>0 or p_len - len(f0) - pad_size>0):
            f0 = np.pad(f0,[[pad_size,p_len - len(f0) - pad_size]], mode='constant')
        f0 *= pow(2, f0_up_key / 12)
        # f0=suofang(f0)
        f0bak = f0.copy()
        f0_coarse=self.get_f0_coarse(f0)
        return f0_coarse, f0bak
    

    def infer(self,audio:np.ndarray,sampling_rate:int) -> np.ndarray:
        '''
        推理函数。
        :param audio: ndarray(n,2)
        :sampling_rate: 采样率
        '''
        
        # f0_up_key=12
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        if sampling_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
            print('test:audio:'+str(audio.shape))
        '''padding'''
        
        
        feats = torch.from_numpy(audio).float()
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
            "source": feats.half().to(device),
            "padding_mask": padding_mask.to(device),
            "output_layer": 9,  # layer 9
        }
        torch.cuda.synchronize()
        with torch.no_grad():
            logits = self.model.extract_features(**inputs)
            feats = self.model.final_proj(logits[0])

        ####索引优化
        npy = feats[0].cpu().numpy().astype("float32")
        D, I = self.index.search(npy, 1)
        # feats = torch.from_numpy(big_npy[I.squeeze()].astype("float16")).unsqueeze(0).to(device)
        index_rate=0.5
        feats = torch.from_numpy(npy).unsqueeze(0).to(device) * index_rate + (1 - index_rate) * feats
        feats=feats.half()

        feats=F.interpolate(feats.permute(0,2,1),scale_factor=2).permute(0,2,1)
        torch.cuda.synchronize()
        # p_len = min(feats.shape[1],10000,pitch.shape[0])#太大了爆显存
        p_len = min(feats.shape[1],12000)#
        pitch, pitchf = self.get_f0(audio, p_len,self.f0_up_key)
        p_len = min(feats.shape[1],12000,pitch.shape[0])#太大了爆显存
        torch.cuda.synchronize()
        # print(feats.shape,pitch.shape)
        feats = feats[:,:p_len, :]
        pitch = pitch[:p_len]
        pitchf = pitchf[:p_len]
        p_len = torch.LongTensor([p_len]).to(device)
        pitch = torch.LongTensor(pitch).unsqueeze(0).to(device)
        pitchf = torch.FloatTensor(pitchf).unsqueeze(0).to(device)
        ii=0#sid
        sid=torch.LongTensor([ii]).to(device)
        with torch.no_grad():
            audio = self.net_g.infer(feats, p_len,pitch,pitchf,sid)[0][0, 0].data.cpu().float().numpy()#nsf
        torch.cuda.synchronize()
        return audio


class Config:
    def __init__(self) -> None:
        self.pth_path:str=''
        self.index_path:str=''
        self.npy_path:str=''
        self.pitch:int=12
        self.samplerate:int=44100
        self.block_time:float=1.0#s
        self.buffer_num:int=1
        self.threhold:int=-30
        self.crossfade_time:float=0.08
        self.extra_time:float=0.04
        self.noise_reduce=False

class GUI:
    def __init__(self) -> None:
        self.config=Config()
        self.flag_vc=False
        
        self.launcher()
    
    def launcher(self):
        sg.theme('LightBlue3')
        input_devices,output_devices,_, _=self.get_devices()
        layout=[
            [
                sg.Frame(title=i18n('加载模型'),layout=[
                    [sg.Input(default_text='TEMP\\atri.pth',key='pth_path'),sg.FileBrowse(i18n('选择.pth文件'))],
                    [sg.Input(default_text='TEMP\\added_IVF512_Flat_atri_baseline_src_feat.index',key='index_path'),sg.FileBrowse(i18n('选择.index文件'))],
                    [sg.Input(default_text='TEMP\\big_src_feature_atri.npy',key='npy_path'),sg.FileBrowse(i18n('选择.npy文件'))]
                ])
            ],
            [
                sg.Frame(layout=[
                    [sg.Text(i18n("输入设备")),sg.Combo(input_devices,key='sg_input_device',default_value=input_devices[sd.default.device[0]])],
                    [sg.Text(i18n("输出设备")),sg.Combo(output_devices,key='sg_output_device',default_value=output_devices[sd.default.device[1]])]
                ],title=i18n("音频设备(请使用同种类驱动)"))
            ],
            [
                sg.Frame(layout=[
                    [sg.Text(i18n("响应阈值")),sg.Slider(range=(-60,0),key='threhold',resolution=1,orientation='h',default_value=-30)],
                    [sg.Text(i18n("音调设置")),sg.Slider(range=(-24,24),key='pitch',resolution=1,orientation='h',default_value=12)]
                    
                ],title=i18n("常规设置")),
                sg.Frame(layout=[
                    [sg.Text(i18n("采样长度")),sg.Slider(range=(0.1,3.0),key='block_time',resolution=0.1,orientation='h',default_value=1.0)],
                    [sg.Text(i18n("淡入淡出长度")),sg.Slider(range=(0.01,0.15),key='crossfade_length',resolution=0.01,orientation='h',default_value=0.08)],
                    [sg.Text(i18n("额外推理时长")),sg.Slider(range=(0.05,3.00),key='extra_time',resolution=0.01,orientation='h',default_value=0.05)],
                    [sg.Checkbox(i18n('输出降噪/Output Noisereduce'),key='noise_reduce')]
                ],title=i18n("性能设置"))
            ],
            [sg.Button(i18n("开始音频转换"),key='start_vc'),sg.Button(i18n("停止音频转换"),key='stop_vc')]
        ]
        
        self.window=sg.Window("RVC - GUI",layout=layout)
        self.event_handler()
    
    def event_handler(self):
        while True:
            event, values = self.window.read()
            if event ==sg.WINDOW_CLOSED:
                self.flag_vc=False
                exit()
            if event == 'start_vc' and self.flag_vc==False:
                self.set_values(values)
                print('pth_path:'+self.config.pth_path)
                print('index_path:'+self.config.index_path)
                print('npy_path:'+self.config.npy_path)
                print('using_cuda:'+str(torch.cuda.is_available()))
                self.start_vc()
            if event=='stop_vc'and self.flag_vc==True:
                self.flag_vc = False


    def set_values(self,values):
        self.set_devices(values["sg_input_device"],values['sg_output_device'])
        self.config.pth_path=values['pth_path']
        self.config.index_path=values['index_path']
        self.config.npy_path=values['npy_path']
        self.config.threhold=values['threhold']
        self.config.pitch=values['pitch']
        self.config.block_time=values['block_time']
        self.config.crossfade_time=values['crossfade_length']
        self.config.extra_time=values['extra_time']
        self.config.noise_reduce=values['noise_reduce']
   
    def start_vc(self):
        torch.cuda.empty_cache()
        self.flag_vc=True
        self.block_frame=int(self.config.block_time*self.config.samplerate)
        self.crossfade_frame=int(self.config.crossfade_time*self.config.samplerate)
        self.sola_search_frame=int(0.012*self.config.samplerate)
        self.delay_frame=int(0.02*self.config.samplerate)#往前预留0.02s
        self.extra_frame=int(self.config.extra_time*self.config.samplerate)#往后预留0.04s
        self.rvc=None
        self.rvc=RVC(self.config.pitch,self.config.pth_path,self.config.index_path,self.config.npy_path)
        self.input_wav:np.ndarray=np.zeros(self.extra_frame+self.crossfade_frame+self.sola_search_frame+self.block_frame)
        self.output_wav:np.ndarray=np.zeros(self.block_frame)
        self.sola_buffer:np.ndarray=np.zeros(self.crossfade_frame,dtype='float32')
        self.fade_in_window:np.ndarray = np.linspace(0, 1, self.crossfade_frame)
        self.fade_out_window:np.ndarray = 1 - self.fade_in_window
        thread_vc=threading.Thread(target=self.soundinput)
        thread_vc.start()


    def soundinput(self):
        '''
        接受音频输入
        '''
        with sd.Stream(callback=self.audio_callback, blocksize=self.block_frame,samplerate=self.config.samplerate,dtype='float32'):
            while self.flag_vc:
                time.sleep(self.config.block_time)
                print('Audio block passed.')
        print('ENDing VC')


    def audio_callback(self,indata:np.ndarray,outdata:np.ndarray, frames, times, status):
        '''
        音频处理
        '''
        start_time=time.perf_counter()
        indata=librosa.to_mono(indata.T)
        self.input_wav[:]=np.roll(self.input_wav,-self.block_frame)
        
        #TODO:Convert all numpy calculation to torch
        '''noise gate'''
        frame_length=1024
        hop_length=512
        rms=librosa.feature.rms(y=indata,frame_length=frame_length,hop_length=hop_length)
        db_threhold=librosa.amplitude_to_db(rms,ref=1.0)[0]<self.config.threhold
        #print(rms.shape,db.shape,db)
        for i in range(db_threhold.shape[0]):
            if db_threhold[i]:
                indata[i*hop_length:(i+1)*hop_length]=0
        self.input_wav[-self.block_frame:]=indata[:]
        
        #infer
        print('input_wav:'+str(self.input_wav.shape))
        infer_wav=librosa.resample(y=self.rvc.infer(self.input_wav[:],self.config.samplerate),orig_sr=40000,target_sr=self.config.samplerate)[-self.crossfade_frame-self.sola_search_frame-self.block_frame:]
        print('infered_wav:'+str(infer_wav.shape))
                
        # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC
        cor_nom = np.convolve(infer_wav[ : self.crossfade_frame + self.sola_search_frame], np.flip(self.sola_buffer), 'valid')
        cor_den = np.sqrt(np.convolve(infer_wav[ : self.crossfade_frame + self.sola_search_frame] ** 2, np.ones(self.crossfade_frame), 'valid') + 1e-3)
        sola_offset = np.argmax( cor_nom / cor_den)
        print('sola offset: ' + str(sola_offset))
        
        # crossfade
        self.output_wav[:]=infer_wav[sola_offset : sola_offset + self.block_frame]
        self.output_wav[:self.crossfade_frame] *= self.fade_in_window
        self.output_wav[:self.crossfade_frame] += self.sola_buffer[:] 
        
        if sola_offset < self.sola_search_frame:
            self.sola_buffer[:] = infer_wav[-self.sola_search_frame - self.crossfade_frame + sola_offset: -self.sola_search_frame + sola_offset]* self.fade_out_window
        else:
            self.sola_buffer[:] = infer_wav[- self.crossfade_frame :]* self.fade_out_window
        
        if self.config.noise_reduce:
            self.output_wav[:]=nr.reduce_noise(y=self.output_wav,sr=self.config.samplerate)
        
        outdata[:]=np.array([self.output_wav,self.output_wav]).T
        print('infer time:'+str(time.perf_counter()-start_time))
        
    def get_devices(self,update: bool = True):
        '''获取设备列表'''
        if update:
            sd._terminate()
            sd._initialize()
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        for hostapi in hostapis:
            for device_idx in hostapi["devices"]:
                devices[device_idx]["hostapi_name"] = hostapi["name"]
        input_devices = [
            f"{d['name']} ({d['hostapi_name']})"
            for d in devices
            if d["max_input_channels"] > 0
        ]
        output_devices = [
            f"{d['name']} ({d['hostapi_name']})"
            for d in devices
            if d["max_output_channels"] > 0
        ]
        input_devices_indices = [d["index"] for d in devices if d["max_input_channels"] > 0]
        output_devices_indices = [
            d["index"] for d in devices if d["max_output_channels"] > 0
        ]
        return input_devices, output_devices, input_devices_indices, output_devices_indices
    
    def set_devices(self,input_device,output_device):
        '''设置输出设备'''
        input_devices,output_devices,input_device_indices, output_device_indices=self.get_devices()
        sd.default.device[0]=input_device_indices[input_devices.index(input_device)]
        sd.default.device[1]=output_device_indices[output_devices.index(output_device)]
        print("input device:"+str(sd.default.device[0])+":"+str(input_device))
        print("output device:"+str(sd.default.device[1])+":"+str(output_device))
        
gui=GUI()
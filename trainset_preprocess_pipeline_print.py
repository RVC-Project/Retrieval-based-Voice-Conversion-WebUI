import sys,os,pdb,multiprocessing
now_dir=os.getcwd()
sys.path.append(now_dir)

inp_root = sys.argv[1]
sr = int(sys.argv[2])
n_p = int(sys.argv[3])
exp_dir = sys.argv[4]
import numpy as np,ffmpeg,os,traceback
from slicer2 import Slicer
from joblib import Parallel, delayed
import librosa,traceback
from  scipy.io import wavfile
import multiprocessing
from my_utils import load_audio
from time import sleep

f = open("%s/preprocess.log"%exp_dir, "a+")
def printt(strr):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()

class PreProcess():
    def __init__(self,sr,exp_dir):
        self.slicer = Slicer(
            sr=sr,
            threshold=-32,
            min_length=800,
            min_interval=400,
            hop_size=15,
            max_sil_kept=150
        )
        self.sr=sr
        self.per=3.7
        self.overlap=0.3
        self.tail=self.per+self.overlap
        self.max=0.95
        self.alpha=0.8
        self.exp_dir=exp_dir
        self.gt_wavs_dir="%s/0_gt_wavs"%exp_dir
        self.wavs16k_dir="%s/1_16k_wavs"%exp_dir
        os.makedirs(self.exp_dir,exist_ok=True)
        os.makedirs(self.gt_wavs_dir,exist_ok=True)
        os.makedirs(self.wavs16k_dir,exist_ok=True)

    def norm_write(self,tmp_audio,idx0,idx1):
        tmp_audio = (tmp_audio / np.abs(tmp_audio).max() * (self.max * self.alpha)) + (1 - self.alpha) * tmp_audio
        wavfile.write("%s/%s_%s.wav" % (self.gt_wavs_dir, idx0, idx1), self.sr, (tmp_audio*32768).astype(np.int16))
        tmp_audio = librosa.resample(tmp_audio, orig_sr=self.sr, target_sr=16000)
        wavfile.write("%s/%s_%s.wav" % (self.wavs16k_dir, idx0, idx1), 16000, (tmp_audio*32768).astype(np.int16))

    def pipeline(self,path, idx0):
        try:
            audio = load_audio(path,self.sr)
            idx1=0
            for audio in self.slicer.slice(audio):
                i = 0
                while (1):
                    start = int(self.sr * (self.per - self.overlap) * i)
                    i += 1
                    if (len(audio[start:]) > self.tail * self.sr):
                        tmp_audio = audio[start:start + int(self.per * self.sr)]
                        self.norm_write(tmp_audio,idx0,idx1)
                        idx1 += 1
                    else:
                        tmp_audio = audio[start:]
                        break
                self.norm_write(tmp_audio, idx0, idx1)
            printt("%s->Suc."%path)
        except:
            printt("%s->%s"%(path,traceback.format_exc()))

    def pipeline_mp(self,infos):
        for path, idx0 in infos:
            self.pipeline(path,idx0)

    def pipeline_mp_inp_dir(self,inp_root,n_p):
        try:
            infos = [("%s/%s" % (inp_root, name), idx) for idx, name in enumerate(sorted(list(os.listdir(inp_root))))]
            ps=[]
            for i in range(n_p):
                p=multiprocessing.Process(target=self.pipeline_mp,args=(infos[i::n_p],))
                p.start()
                ps.append(p)
            for p in ps:p.join()
        except:
            printt("Fail. %s"%traceback.format_exc())

if __name__=='__main__':
    # f = open("logs/log_preprocess.log", "w")
    printt(sys.argv)
    ######################################################
    # inp_root=r"E:\语音音频+标注\米津玄师\src"
    # inp_root=r"E:\codes\py39\vits_vc_gpu_train\todo-songs"
    # sr=40000
    # n_p = 6
    # exp_dir=r"E:\codes\py39\dataset\mi-test"

    ######################################################
    printt("start preprocess")
    pp=PreProcess(sr,exp_dir)
    pp.pipeline_mp_inp_dir(inp_root,n_p)
    printt("end preprocess")

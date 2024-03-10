name="YourName" # Output File will be <YourName-FileName>.mp3
model_name="js-20231220.pth" # model in the path assets/weights/
file_index="logs/js-20231220/trained_IVF358_Flat_nprobe_1_js-20231220_v2.index" # index file in the path logs/<modelName>/
save_root_path="C:/Users/Conda/Desktop/AI_data/audio/result"  
'''
使用这个脚本是为了批量转换到单个角色的音频，可以自动进行音轨分离和合并。输入文件支持文件夹和单文件。
Use this script to batch convert audio to a single character, allowing automatic track separation and merging. Input files support folders and single files.
Create a bat file to run this script.
@echo off
cd C:\Users\Conda\Desktop\dev\RVC1006Nvidia
runtime\python.exe batch-example.py --pycmd runtime\python.exe
pause
'''
import os, sys, shutil, subprocess
from dotenv import load_dotenv
import soundfile as sf
now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()
from infer.modules.vc.modules import VC
from configs.config import Config
from infer.modules.uvr5.modules import uvr

# 底下的没必要改
f0_file="assets/weights/"+model_name
file_index2=[]
f0_method="rmvpe" 
index_rate=0.75 # 检索特征占比, 0-1
filter_radius=3 #>=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音, 0-7
resample_sr=0 # 后处理重采样至最终采样率，0为不进行重采样, 0-48000
rms_mix_rate=0.25 # 输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络,0-1
protect=0.33 # 保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果, 0-0.5

def mix_audio(path_accompaniment, path_vocal, output_name): # 使用ffmpeg合并伴奏和人声
    command = [
        'ffmpeg',
        '-i', path_accompaniment,
        '-i', path_vocal,
        '-filter_complex', '[1:a]volume=1.5[a1];[0:a][a1]amix=inputs=2:duration=first:dropout_transition=3',
        output_name
    ]
    try:
        subprocess.run(command, check=True)
        print(f"Audio mixed successfully into {output_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error during audio mixing: {e}")

def RVC(input_audio, vc_transform=0, output_audio = "output_audio.wav"):
    config = Config()
    vc = VC(config)
    fn=vc.get_vc(model_name)
    result = vc.vc_single(0, input_audio, vc_transform,  # 变调(整数, 半音数量, 升八度12降八度-12)
                          f0_file, f0_method, file_index, file_index2,
                          index_rate, filter_radius, 
                          resample_sr, rms_mix_rate, protect)
    
    print("Inference result:", result)
    if result and len(result) > 1 and result[1] is not None:
        audio_data = result[1][1]
        sample_rate = 40000
        sf.write(output_audio, audio_data, sample_rate)
        print(f"Audio data saved to {output_audio}")
    else:
        print("Inference result:", result)

def splitTrack(music_path, save_root_vocal, save_root_ins):
    model_name="HP5_only_main_vocal"
    dir_wav_input=""
    agg=10 #人声提取激进程度, 0-20
    format0="flac" # 导出文件格式, ["wav", "flac", "mp3", "m4a"]
    paths=[]
    inp_root=music_path
    generator = uvr(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0)
    # 迭代生成器以获取并处理每个信息
    for result in generator:
        print(result)
    return True

def find_files(directory, extensions):
    """Traverse a directory and find files with the given extensions."""
    files_found = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.split('.')[-1].lower() in extensions:
                files_found.append(os.path.join(root, file))
    return files_found

def extract_filename_without_extension(file_path):
    """
    Extract the filename without its extension from a full file path.

    Args:
    file_path (str): The full path to the file.

    Returns:
    str: The filename without its extension.
    """
    filename_with_extension = os.path.basename(file_path)
    filename_without_extension, _ = os.path.splitext(filename_with_extension)
    return filename_without_extension
    
def create_directory_if_not_exists(directory_path):
    """
    Create a directory if it does not exist.

    Args:
    directory_path (str): The path to the directory to create.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")

def delete_directory(directory_path):
    try:
        shutil.rmtree(directory_path)
        print(f"Directory '{directory_path}' and all its contents have been deleted.")
    except OSError as error:
        print(f"Error: {error}. Failed to delete '{directory_path}'.")

def main(input_audio,tone=0):
    global isMusic, save_path
    if isMusic:
        save_path=os.path.join(save_root_path,"music/")
        music_name=extract_filename_without_extension(input_audio)
        save_path=os.path.join(save_path,music_name)
        source_music_path=os.path.join(save_path,"source/")
        tmp_path=os.path.join(save_path,"tmp")
        source_music=os.path.join(source_music_path,os.path.basename(input_audio))
        result_path=os.path.join(save_path,"result/")
        vocal_path=os.path.join(save_path,"vocal/")
        result_vocal_path=os.path.join(vocal_path,f"{name}_vocal_{music_name}.wav")
        final_name=h=os.path.join(result_path,f"{name}_{music_name}.mp3")
        create_directory_if_not_exists(save_path)
        create_directory_if_not_exists(input_audio)
        create_directory_if_not_exists(source_music_path)
        create_directory_if_not_exists(tmp_path)
        create_directory_if_not_exists(vocal_path)
        create_directory_if_not_exists(result_path)
        print(f"""音乐文件模式\n输入文件：{input_audio}, 保存文件根目录: {save_path},
复制原音乐保存到：{source_music_path}, 待分离音轨的音乐: {source_music}, 
最终文件输出目录: {result_path}, result_vocal_path: {result_vocal_path}, final_name: {final_name}""")
        shutil.copy(input_audio, source_music_path)
        shutil.copy(input_audio, tmp_path)
        path_vocal=""; path_instrument=""
        for i in os.listdir(source_music_path):
            if "vocal" in i:
                path_vocal=os.path.join(source_music_path, i)
            if "instrument" in i:
                path_instrument=os.path.join(source_music_path, i)
        if path_vocal!="" and path_instrument!="":
            print(f"分离后的音轨文件已经存在: 人声路径: {path_vocal}, 伴奏路径: {path_instrument}")
        else:
            splitTrack(music_path=tmp_path, save_root_vocal=source_music_path, save_root_ins=source_music_path)
            for i in os.listdir(source_music_path):
                if "vocal" in i:
                    path_vocal=os.path.join(source_music_path, i)
                if "instrument" in i:
                    path_instrument=os.path.join(source_music_path, i)
            print(f"音轨分离完成: 人声路径: {path_vocal}, 伴奏路径: {path_instrument}")
        RVC(path_vocal, vc_transform=tone, output_audio = result_vocal_path)
        delete_directory(tmp_path)
        mix_audio(path_accompaniment=path_instrument, path_vocal=result_vocal_path, output_name=final_name)
    else:
        save_path=os.path.join(save_root_path,"basic",name)
        save_audio_path=os.path.join(save_path,extract_filename_without_extension(input_audio)+".wav")
        create_directory_if_not_exists(save_path)
        print(save_audio_path)
        RVC(input_audio, vc_transform=tone, output_audio = save_audio_path)

while True:
    input_audio=input("你想处理的文件或者是文件夹(注意如果是目录的话会遍历子目录)：")
    if os.path.exists(input_audio):
        break
    else:
        print(f"文件或文件夹不存在: {input_audio}")
while True:
    tone=input("变调(整数, 半音数量, 升八度12降八度-12), 必须为整数, 默认为0，留空为默认: ")
    if tone.replace(" ","")=="":
        tone=0
        break
    else:
        try:
            tone=int(tone)
            break
        except Exception as e:
            print(f"变调参数错误：{e}")
            pass
isMusic=input("是不是音乐文件\n(True 的情况下，会在<save_path>/music文件夹下创建<音乐名>文件夹，自动分离音轨，自动转换，自动合成\nFalse 的情况下，文件会保存到<save_path>/<basic>/<name>文件夹下\n留空默认为False, 即不是音乐\n:")
if isMusic.replace(" ","")=="":
    isMusic==False
else:
    isMusic==True
if os.path.isdir(input_audio)==True:
    files=find_files(directory=input_audio, extensions= ["wav", "flac", "mp3", "m4a"])
    for i in files:
        print(i)
        main(i,tone)
else:
    main(input_audio,tone)
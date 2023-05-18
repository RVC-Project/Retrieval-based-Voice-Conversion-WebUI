import os

os.chdir('logs/ai-zzzyt/')
for i in os.listdir():
    if i.endswith('.pt.trace.json'):
        s=open(i,encoding='utf-8').read()
        s=s.replace('\\','/')
        open(i,'w',encoding='utf-8').write(s)
                  
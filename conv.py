import os

# os.chdir('logs/ai-zzzyt/')
def proc(d):
    print('CONV:',d)
    for i in os.listdir(d):
        if i.endswith('.pt.trace.json'):
            s=open(i,encoding='utf-8').read()
            s=s.replace('\\','/')
            open(i,'w',encoding='utf-8').write(s)
proc('.')
proc('logs/ai-zzzyt')
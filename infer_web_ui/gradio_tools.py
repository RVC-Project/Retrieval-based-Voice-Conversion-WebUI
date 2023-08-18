from time import sleep

# Constants
sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


# Functions
def clean():
    return {"value": "", "__type__": "update"}


def if_done(done, p):
    while 1:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True


def if_done_multi(done, ps):
    while 1:
        flag = 1
        for p in ps:
            if p.poll() is None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True

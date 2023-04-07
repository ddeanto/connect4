from multiprocessing import Pool
import time
from pydantic import BaseModel

class InputF(BaseModel):
    x: float
    y: float


def f(v: InputF):
    time.sleep(1)
    x = v.x
    y = v.y
    return x*y

if __name__ == '__main__':
    begin = time.time()
    with Pool(5) as p:
        out = p.map(f, [InputF(x=2,y=3), InputF(x=4.3,y=3), InputF(x=2,y=3.3), InputF(x=342,y=4.3), InputF(x=872,y=6), InputF(x=76,y=67), InputF(x=256.6,y=5.6)] )
    end = time.time()
    print(round(end-begin, 1))
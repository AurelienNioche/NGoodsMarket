import multiprocessing
import os
import numpy as np
from tqdm import tqdm


def _run(x):
    multiprocessing.Event().wait(np.random.random())
    return x


with multiprocessing.Pool(processes=os.cpu_count() - 1) as p:
    max_ = 30
    with tqdm(total=max_) as pbar:
        for i, x in tqdm(enumerate(p.imap_unordered(_run, range(0, max_)))):
            pbar.update()
            print(i, x)



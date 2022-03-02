from lauges_tqdm import tqdm

import time
for i in tqdm(range(5)):
    for j in tqdm(range(13),1):
        for j in tqdm(range(10),1):
            # for j in tqdm(range(3),1):
                time.sleep(0.001)
for i in tqdm(range(5)):
    for j in tqdm(range(13),1):
        # for j in tqdm(range(10),1):
            # for j in tqdm(range(3),1):
                time.sleep(0.001)

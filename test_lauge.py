from lauges_tqdm import tqdm

import time
for i in tqdm(5):
    for j in tqdm(13):
        for j in tqdm(10):
            # for j in tqdm(range(3),1):
                time.sleep(0.001)
for i in tqdm(5,17):
    for j in tqdm(13):
        # for j in tqdm(range(10),1):
            # for j in tqdm(range(3),1):
                time.sleep(0.001)

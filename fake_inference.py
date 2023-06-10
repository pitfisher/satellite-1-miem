import sys
from pathlib import Path
import random

import pandas as pd

random.seed(41)

TEST_IMAGES_PATH, SAVE_PATH = sys.argv[1:]


def generate_random_solution(f):
    xc = random.random() * 0.8
    yc = random.random() * 0.8
    w = random.random() * 0.2
    h = random.random() * 0.2
    conf = random.random()
    image_id = f.name[:-len(f.suffix)]

    result = {
        'image_id': image_id,
        'xc': round(xc, 4),
        'yc': round(yc, 4),
        'w': round(w, 4),
        'h': round(h, 4),
        'label': 0,
        'score': round(conf, 4)
    }
    return result


def create_simple_solution():
    results = []
    for f in Path(TEST_IMAGES_PATH).glob('*.JPG'):
        if random.random() > 0.7:
            for _ in range(10):
                if random.random() > 0.6:
                    results.append(generate_random_solution(f))

    test_df = pd.DataFrame(results, columns=['image_id', 'xc', 'yc', 'w', 'h', 'label', 'score'])
    test_df.to_csv(SAVE_PATH, index=False)


def main():
    create_simple_solution()


if __name__ == '__main__':
    main()

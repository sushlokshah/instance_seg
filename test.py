import numpy as np

for m in range(1, 10):
    random_color = np.array(
                [10000/(m**2), m**m, 5000/m]
            )
    random_color = np.random.randint(0, 255, size=3)
            # print(random_color)
    random_color = random_color.astype(np.uint8)
    print(random_color)
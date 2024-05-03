import random


def generate_unit_square(size=4, iterations=200000, type='train'):
    unit_square = [(1, 1)] * (size*size*iterations)
    with open(f'./data/{type}_unit_square_{size}_{iterations}.txt', 'w') as f:
        for w,h in unit_square:
            f.write(f'{w} {h}\n')


def generate_rectangular(size=4, iterations=300000, type='train'):
    rectangles = [(random.randint(1, size), random.randint(1, size)) 
                  for _ in range(size * size * iterations)]
    with open(f'./data/{type}_rectangular_{size}_{iterations}.txt', 'w') as f:
        for w,h in rectangles:
            f.write(f'{w} {h}\n')


def generate_square(size=4, iterations=300000, type='train'):
    square = []
    for _ in range(iterations):
        for _ in range(size*size):
            Iwt = random.randint(1, size)  # 將隨機生成的浮點數轉換為整數
            square.append((Iwt, Iwt))
    with open(f'./data/{type}_square_{size}_{iterations}.txt', 'w') as f:
        for w,h in square:
            f.write(f'{w} {h}\n')
    

if __name__ == '__main__':
    random.seed(777)
    
    '''tarin data'''
    # generate_unit_square(size=3, iterations=200000, type='train')
    generate_unit_square(size=4, iterations=200000, type='train')
    generate_unit_square(size=5, iterations=300000, type='train')

    # generate_rectangular(size=3, iterations=300000, type='train')
    generate_rectangular(size=4, iterations=300000, type='train')
    generate_rectangular(size=5, iterations=350000, type='train')

    # generate_square(size=3, iterations=300000, type='train')
    generate_square(size=4, iterations=300000, type='train')
    generate_square(size=5, iterations=350000, type='train')

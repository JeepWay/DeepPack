import random

def generate_unit_square(size=4, iterations=200000, type='train'):
    unit_square = [(1, 1)] * (size*size*iterations)
    with open(f'./data/{type}_unit_square_{size}_random_{iterations}.txt', 'w') as f:
        for w,h in unit_square:
            f.write(f'{w} {h}\n')


def generate_rectangular(size=4, iterations=300000, type='train'):
    rectangles = [(random.randint(1, size), random.randint(1, size)) 
                  for _ in range(size * size * iterations)]
    with open(f'./data/{type}_rectangular_{size}_random_{iterations}.txt', 'w') as f:
        for w,h in rectangles:
            f.write(f'{w} {h}\n')


def generate_square(size=4, iterations=300000, type='train'):
    square = []
    for _ in range(iterations):
        for _ in range(size*size):
            Iwt = random.randint(1, size)  # 將隨機生成的浮點數轉換為整數
            square.append((Iwt, Iwt))
    with open(f'./data/{type}_square_{size}_random_{iterations}.txt', 'w') as f:
        for w,h in square:
            f.write(f'{w} {h}\n')
    

def generate_sequences(bin_type:str, size, type:str, iterations, sequence_composition):
    sequences = []
    for _ in range(iterations):
        sequence = []
        for frequence, item in sequence_composition:
            for _ in range(frequence):
                sequence.append(item)
        random.shuffle(sequence)  # Shuffle the sequence to ensure randomness
        sequences.append(sequence)
    # save to txt
    with open(f'./data/test_{bin_type}_{size}_{type}_{iterations}.txt', 'w') as f:
        for seq in sequences:
            for w,h in seq:
                f.write(f'{w} {h}\n')
    return sequences


if __name__ == '__main__':
    random.seed(666)

    '''test data'''
    # generate_unit_square(size=3, iterations=20000, type='test')
    # generate_unit_square(size=4, iterations=20000, type='test')
    # generate_unit_square(size=5, iterations=20000, type='test')

    # generate_rectangular(size=3, iterations=20000, type='test')
    generate_rectangular(size=4, iterations=20000, type='test')
    generate_rectangular(size=5, iterations=20000, type='test')

    # generate_square(size=3, iterations=20000, type='test')
    generate_square(size=4, iterations=20000, type='test')
    generate_square(size=5, iterations=20000, type='test')


    # per-defined sequences
    unit_square_4x4_composition = {
        'type1': [(16, (1, 1))],
    }

    square_4x4_composition = {
        'type1': [(1, (4, 4))],
        'type2': [(1, (3, 3)), (7, (1, 1))],
        'type3': [(3, (2, 2)), (4, (1, 1))]
    }
    
    rectangular_4x4_composition = {
        'type1': [(1, (3, 2)), (1, (1, 4)), (1, (2, 2)), (1, (1, 2))],
        'type2': [(1, (1, 1)), (1, (1, 3)), (1, (3, 1)), (1, (3, 3))],
        'type3': [(1, (4, 2)), (2, (2, 2))]
    }

    unit_square_5x5_composition = {
        'type1': [(25, (1, 1))],
    }

    square_5x5_composition = {
        'type1': [(1, (5, 5))],
        'type2': [(4, (2, 2)), (9, (1, 1))],
        'type3': [(1, (3, 3)), (3, (2, 2)), (4, (1, 1))]
    }

    rectangular_5x5_composition = {
        'type1': [(4, (2, 2)), (1, (1, 4)), (1, (4, 1)), (1, (1, 1))],
        'type2': [(1, (2, 5)), (1, (3, 3)), (1, (3, 2))],
        'type3': [(1, (4, 4)), (1, (1, 4)), (1, (4, 1)), (1, (1, 1))]
    }

    for i in range(len(unit_square_4x4_composition)):
        generate_sequences('unit_square', 4, f'type{i+1}', 5000, unit_square_4x4_composition[f'type{i+1}'])

    for i in range(len(square_4x4_composition)):
        generate_sequences('square', 4, f"type{i+1}", 5000, square_4x4_composition[f'type{i+1}'])

    for i in range(len(rectangular_4x4_composition)):
        generate_sequences('rectangular', 4, f"type{i+1}", 5000, rectangular_4x4_composition[f'type{i+1}'])

    for i in range(len(unit_square_5x5_composition)):
        generate_sequences('unit_square', 5, f'type{i+1}', 5000, unit_square_5x5_composition[f'type{i+1}'])

    for i in range(len(square_5x5_composition)):
        generate_sequences('square', 5, f"type{i+1}", 5000, square_5x5_composition[f'type{i+1}'])

    for i in range(len(rectangular_5x5_composition)):
        generate_sequences('rectangular', 5, f"type{i+1}", 5000, rectangular_5x5_composition[f'type{i+1}'])

        
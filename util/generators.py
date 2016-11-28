import MeCab
import os

def batch(generator, batch_size):
    """
    call the batch function
    :param generator: make the bath data
    :param batch_size: setting the batch size
    :return: batch tuple
    """
    batch = []
    is_tuple = False
    for l in generator:
        is_tuple = isinstance(l, tuple)
        batch.append(l)
        if len(batch) == batch_size:
            yield tuple(list(x) for x in zip(*batch)) if is_tuple else batch
            batch = []
    if batch:
        yield tuple(list(x) for x in zip(*batch)) if is_tuple else batch

def sorted_parallel(generator1, generator2, pooling, order=1):
    """
    sort parallerl source and target
    :param generator1:
    :param generator2:
    :param pooling:
    :param order:
    :return:
    """
    gen1 = batch(generator1, pooling)
    gen2 = batch(generator2, pooling)
    for batch1, batch2 in zip(gen1, gen2):
        yield from sorted(zip(batch1, batch2), key=lambda x: len(x[1]))
        for x in sorted(zip(batch1, batch2), key=lambda x: len(x[order])):
            yield x

def word_list(filename):
    with open(filename, 'r', encoding='utf-8') as fp:
        for l in fp:
            yield l.split()

def letter_list(filename):
    with open(filename) as fp:
        for l in fp:
            yield list(''.join(l.split()))

def input_file(filename):

    mecab = MeCab.Tagger("-Owakati")

    try:
        src = input('> ')

        if src == 'exit':
            exit()
        if src in ['rm', 'reset']:
            os.remove(filename)

        src = mecab.parse(src)

    except Exception as e:
        print(e.message)

    f = open(filename, 'a+')
    try:
        f.write(src)
    finally:
        f.close()


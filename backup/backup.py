import pickle


def save(obj, file):
    pickle.dump(obj=obj, file=open(f'data/{file}', 'wb'))


def load(file):
    return pickle.load(file=open(f'data/{file}', 'rb'))

import pickle
import os

folder = 'data'


def save(obj, file):
    os.makedirs(folder, exist_ok=True)
    pickle.dump(obj=obj, file=open(f'{folder}/{file}', 'wb'))


def load(file):
    print(f"Loading file '{file}'...")
    return pickle.load(file=open(f'{folder}/{file}', 'rb'))

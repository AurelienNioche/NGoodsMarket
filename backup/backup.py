import pickle
import os


def save(obj, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    print(f"Loading file '{file_name}'...")
    pickle.dump(obj=obj, file=open(file_name, 'wb'))


def load(file_name):
    print(f"Loading file '{file_name}'...")
    return pickle.load(file=open(file_name, 'rb'))

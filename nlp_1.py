#https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
import string
import unicodedata
import torch

allowed_chars = string.ascii_letters + " .,;'"+ "_"
n_letters = len(allowed_chars)
print(n_letters)

# thanks to thanks to https://stackoverflow.com/a/518232/2809427

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD',s)
        if unicodedata.category(c) != 'Mn'
        and c in allowed_chars
    )

print(f"Convering 'ä, ö, and ü' to {unicodeToAscii('ä, ö, and ü')} ")

# making tensors
# using batch size of 1

def letterToIndex(letter):
    if letter not in allowed_chars:
        return allowed_chars.find("_")
    else:
        return allowed_chars.find(letter)

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

print(f'Letter j index {letterToIndex('j')}')
print(f"letter j in a line {lineToTensor('j')}")













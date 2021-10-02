import pandas as pd

file = open("text.txt", "rb")

sentences = file.readlines()
print(sentences[1:10])
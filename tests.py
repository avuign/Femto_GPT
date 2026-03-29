from data import Tokenizer, dic, inputs_and_targets, text_to_words

with open("alice.txt") as file:
    text = file.read()

tokenizer = Tokenizer(dic(text_to_words(text)))

print(len(tokenizer.word_to_int))

code = tokenizer.encode("this moment the door of the house opened with AJ")
print(code)

text = tokenizer.decode(code)
print(text)

print(inputs_and_targets(text, 3))

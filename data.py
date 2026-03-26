def encoding_dic(words):
    encoding = {word: i for i, word in enumerate(list(set(words)))}
    decoding = {i: word for word, i in encoding.items()}
    return encoding, decoding


def text_to_input(words, context_size):
    contexts = []
    targets = []
    dic, _ = encoding_dic(words)
    for i in range(0, len(words) - context_size):
        window = words[i : i + context_size]
        target = words[i + context_size]

        context = []
        for word in window:
            context.append(dic[word])

        contexts.append(context)
        targets.append(dic[target])
    return contexts, targets


def load_data(filename, context_size):
    with open(filename) as text:
        words = text.read().split()
    input_context, targets = text_to_input(words, context_size)
    return input_context, targets

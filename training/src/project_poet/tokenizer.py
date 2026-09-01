
def get_tokenizer(raw_data):
    corpus = raw_data["Title"].str.cat() + raw_data["Poem"].str.cat()
    characters = sorted(list(set(corpus)))
    stoi = {s: i for i, s in enumerate(characters)}
    itos = {i: s for i, s in enumerate(characters)}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda i: "".join([itos[c] for c in i])

    return encode, decode

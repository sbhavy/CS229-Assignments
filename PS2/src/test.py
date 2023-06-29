def get_words(message):
    words = message.split()
    for i in range(len(words)):
        words[i] = words[i].lower()
    return words

messages = ["hello mic mic hello mic", "hello mic is bhavyesh"]

index = 0
word_dict = {}
for message in messages:
    words = get_words(message)
    for word in list(set(words)):
        if word not in word_dict:
            word_dict[word] = [index, 1]
            index += 1
        else:
            word_dict[word][1] += 1

deleters = []        

for i in word_dict:
    if word_dict[i][1] >= 2: word_dict[i] = word_dict[i][0]  
    else: deleters.append(i)

for i in deleters:
    del word_dict[i]

print(word_dict)
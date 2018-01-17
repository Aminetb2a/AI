# Ahmed Amine TALEB BAHMED
# 130201120


from math import log10

def read_corpus(corpus_file):
    out = []
    with open(corpus_file) as f:
        for line in f:
            tokens = line.strip().split()
            out.append( (tokens[1], tokens[3:]) )
    return out

all_docs = read_corpus('all_sentiment_shuffled.txt')
split_point = int(0.8*len(all_docs))
train_docs = all_docs[:split_point]
eval_docs = all_docs[split_point:]

bag = {}		# "word_i":[word counts in bag, word counts in +, p(w|+), p(w|-)]
docs = [0, 0]	# [total # reviews, # positive reviews]
total = [0,0,0]	# [total # words in vocabulary, total # words in positive reviews, total # in negative]
ppos = 0		# P(+) = 1 - p(-)
k = 10           #as k goes grater the probabilities go closer
accuracy = [0,0]# [correct, incorrect]

def train_nb(train_docs):
    global bag
    global docs
    global total
    global ppos
    global k

    for line in train_docs[:-1]:

        docs[0] += 1
        document = negation(line[1])
        # document = line[1]

        for i in range (len(document)):
            word = document[i]
            if len(word) < 4:
                continue
			# if word is not in bag, add it
            if word.lower() not in bag:
                bag[word.lower()] = [1, 0, 0, 0]

            else:
                bag[word.lower()][0] += 1

            if (line[-2]) == 'pos':
                bag[word.lower()][1] += 1

		#if it is a word in a positive review, increase # positive words
        if (line[-2])== 'pos':
            docs[1] += 1

        total[0] += 1
		# update word counts
        if (line[-2])=='pos':
            total[1] += 1
        else:
            total[2] += 1

	# calculate probabilities
    ppos = (docs[1] + k) / (docs[0] + k*2)

    # calculate P(word|+), p(word|-)
    for word in bag:
        bag[word][2] = (bag[word][1] + k) / (total[1] + k*len(bag))
        bag[word][3] = (bag[word][0] - bag[word][1] + k) / (total[2] + k*len(bag))

    return bag

def classify_nb(classifier_data, document):

    positive = log10(ppos)
    negative = log10(1 - ppos)
    bag = classifier_data

    for i in range (len(document)):

        word = document[i]
        if word in bag:
            positive += log10(bag[word][2])
            negative += log10(bag[word][3])
        else:
             positive += log10((k / (total[1] + k*len(bag)) ))
             negative += log10((k / (total[2] + k*len(bag)) ))

    if positive > negative:
        return "pos"
    else:
        return "neg"


def evaluate_nb (classifier_data, evaluation_documents):

    for line in evaluation_documents[:-1]:
        document = negation(line[1])
        # document = line[1]
        label = classify_nb(classifier_data, document)

        if label == line[0]:
            accuracy[0] += 1
        else:
            accuracy[1] += 1

    acuracy = accuracy[0]*1.0/(accuracy[0]+accuracy[1])

    return acuracy

def negation(text):

    words = text
    l = len(words)
    j=1

    for i in range(l):
        if (i < (l-j)):
            if( words[i]=="not" or words[i].find("n't")>=0):
                words[i] = words[i] +" "+words[i+1]
                words.remove(words[i+1])
                j+=1
    return words

# print(evaluate_nb(train_nb(train_docs),eval_docs))

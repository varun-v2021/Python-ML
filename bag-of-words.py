# https://www.freecodecamp.org/news/an-introduction-to-bag-of-words-and-how-to-code-it-in-python-for-nlp-282e87a9da04/
# John likes to watch movies. Mary likes movies too.[1, 2, 1, 1, 2, 1, 1, 0, 0, 0]
# John - 1, likes - 2, to - 1, watch - 1, movies - 2
# John also likes to watch football games.[1, 1, 1, 1, 0, 0, 0, 1, 1, 1]

# BOW = clean text -> tokenize -> build vocabulary -> generate vectors

# from both above arrays, frequency of the entire document is

# {"John":2,"likes":3,"to":2,"watch":2,"movies":2,"Mary":1,"too":1,"also":1,"football":1,"games":1}

# The length of the vector will always be equal to vocabulary size. In this case the vector length is 10.
# In order to represent our original sentences in a vector, each vector is initialized with all zeros —
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Stopwords are words which do not contain enough significance to be used without our algorithm.
# Tokenization is the act of breaking up a sequence of strings into pieces such as words, keywords, phrases, symbols and other elements called tokens.
import re
import numpy

# INPUT ARRAY
# ["Joe waited for the train", "The train was late", "Mary and Samantha took the bus",
# "I looked for Mary and Samantha at the bus station",
# "Mary and Samantha arrived at the bus station early but waited until noon for the bus"]

# STEP 1
def word_extraction(sentence):
    ignore = ['a', 'the', 'is'] # Stopwords
    # regex.sub will replace [^\w] left most occurence with " " in sentence
    # split() with no parameter, sentence will be separated by whitespaces which includes \n and tabs \t
    # words will have an array of tokens
    words = re.sub("[^\w]", " ",  sentence).split()
    # convert to lower case and filter stopwords
    cleaned_text = [w.lower() for w in words if w not in ignore]
    return cleaned_text
# Using nltk above implementation is built-in
# import nltkfrom nltk.corpus import stopwords set(stopwords.words('english'))

# STEP 2
#  Apply tokenization to all sentences
def tokenize(sentences):
    # collect all the words from all the sentences
    words = []
    for sentence in sentences:
        w = word_extraction(sentence)
        # collect all the words from each sentence
        words.extend(w)
    words = sorted(list(set(words)))
    # ['and', 'arrived', 'at', 'bus', 'but', 'early', 'for', 'i', 'joe', 'late', 'looked', 'mary', 'noon', 'samantha', 'station', 'the', 'took', 'train', 'until', 'waited', 'was']
    return words

# STEP 3
# Build vocabulary and generate vectors
def generate_bow(allsentences):
    vocab = tokenize(allsentences)
    print("Word List for Document \n{0} \n".format(vocab))
    for sentence in allsentences:
        words = word_extraction(sentence)
        # initialize all words are having 0 count in bag_vector
        bag_vector = numpy.zeros(len(vocab))
        for w in words:
            for i,word in enumerate(vocab):
                # if word in vocab is found in extracted words, then increment its count in bag_vector
                if word == w:
                    # i represents the position of the word in the sentence
                    bag_vector[i] += 1
        print("{0}\n{1}\n".format(sentence, numpy.array(bag_vector)))

# for learning about extend functionality
def extend_func_example():
    my_list = ['geeks', 'for']
    another_list = [6, 0, 4, 1]
    my_list.extend(another_list)
    print(my_list)
    # ['geeks', 'for', 6, 0, 4, 1]
    my_list = ['geeks', 'for', 6, 0, 4, 1]
    my_list.extend('geeks')
    print(my_list)
    # ['geeks', 'for', 6, 0, 4, 1, 'g', 'e', 'e', 'k', 's']

# for learning about enumerate functionality
# enumerate function keeps a count of the iteration
def enumerate_func_example():
    lst = ['eat','sleep','repeat']
    s1 = 'geek'
    obj1 = enumerate(lst)
    obj2 = enumerate(s1,2)
    print(obj1)
    print(obj2)
    # [(0, 'eat'), (1, 'sleep'), (2, 'repeat')]
    # [(2, 'g'), (3, 'e'), (4, 'e'), (5, 'k')]

if __name__ == "__main__":
    allsentences = ["Joe waited for the train train", "The train was late", "Mary and Samantha took the bus",
                    "I looked for Mary and Samantha at the bus station",
                    "Mary and Samantha arrived at the bus station early but waited until noon for the bus"]
    generate_bow(allsentences)

# As you can see, each sentence was compared with our word list generated in Step 1.
# Based on the comparison, the vector element value may be incremented.
# These vectors can be used in ML algorithms for document classification and predictions.
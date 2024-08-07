import re


# function for removing the stopwords from the text
def remove_stopwords(text, stopwords):
    return " ".join([word for word in str(text).split() if word not in stopwords])

# function for removing any punctuation signs from the text
def remove_punctuations(text, punctuations):
    translator = str.maketrans('', '', punctuations)
    return text.translate(translator)

# function for removing the URL addresses
def remove_URLs(text):
    return re.sub('((www[^s]+)|(http[^s]+)|(https[^s]+))', ' ', text)

# function for removing the numeric numbers
def remove_numbers(text):
    return re.sub('[0-9]+', ' ', text)

# function for stemming the text
def stemming(data, st):
    text = [st.stem(word) for word in data]
    return data

# function for lemmatization the text
def lemmatizing(data, lm):
    text = [lm.lemmatize(word) for word in data]
    return data


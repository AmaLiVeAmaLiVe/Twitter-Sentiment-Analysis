import string
import unittest
import nltk
import stopwords
import sys

sys.path.append('../scripts')

from preprocessing_data import remove_stopwords, remove_punctuations, remove_URLs, remove_numbers, stemming, lemmatizing

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.text = "This is a simple example of the tweet! Visit https://example.com for more info. Call us at 123-456-7890."
        self.words = ["running", "jumps", "easily", "fairly"]
        self.stopwords = stopwords.get_stopwords("english")
        self.punctuations = string.punctuation
        self.stemmer = nltk.PorterStemmer()
        self.lm = nltk.WordNetLemmatizer()

    def test_remove_stopwords(self):
        result = remove_stopwords(self.text, self.stopwords)
        expected = "This simple example tweet! Visit https://example.com info. Call us 123-456-7890."
        self.assertEqual(result, expected)

    def test_remove_punctuations(self):
        result = remove_punctuations(self.text, self.punctuations)
        expected = "This is a simple example of the tweet Visit httpsexamplecom for more info Call us at 1234567890"
        self.assertEqual(result, expected)

    def test_remove_URLs(self):
        result = remove_URLs(self.text)
        expected = "This is a simple example of the tweet! Visit  s at 123-456-7890."
        self.assertEqual(result, expected)

    def test_remove_numbers(self):
        result = remove_numbers(self.text)
        expected = "This is a simple example of the tweet! Visit https://example.com for more info. Call us at  - - ."
        self.assertEqual(result, expected)

    def test_stemming(self):
        result = stemming(self.words, self.stemmer)
        expected = ['running', 'jumps', 'easily', 'fairly']
        self.assertEqual(result, expected)

    def test_lemmatizer(self):
        result = lemmatizing(self.words, self.lm)
        expected = ["running", "jumps", "easily", "fairly"]
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()



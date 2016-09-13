#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
import string


def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """

    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()
    # initiate stemmer:
    stemmer = SnowballStemmer('english')
    # split off metadata
    content = all_text.split("X-FileName:")
    words = ''
    output = ''
    if len(content) > 1:
        # remove punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)
        words = text_string.replace('\t', " ").replace('\n', " ").replace('\r', " ")
        test = [var for var in words.split(" ") if var != '']
        for word in test:
            sample = stemmer.stem(word)
            output += sample + " "
            # split the text string into individual words, stem each word,
            # and append the stemmed word to words (make sure there's a single
            # space between each stemmed word)

    return output


def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print text
    ff.close()


if __name__ == '__main__':
    main()

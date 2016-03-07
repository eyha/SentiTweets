import re

def findEmotes(wordBag):
    # match all tokens that have consecutive special characters
    # - can't handle ones containing whitespace, as tokens are separated by whitespace anyway
    # - ignores examples beginning with #, as these are assumed to be hashtags 
    nonword = re.compile('[^0-9a-zA-Z#][^0-9a-zA-Z]+')
    emoticons = []
    for word in wordBag:
        if nonword.match(word):
            emoticons.append(word)
    return emoticons
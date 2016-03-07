import re

def findEmotes(wordBag):
    # match all tokens that have more than one special characters - need to handle HTML special characters
    # - can't handle ones containing whitespace, as tokens are separated by whitespace anyway
    # - ignores examples beginning with #, as these are assumed to be hashtags 
    nonword = re.compile('((&(.*);)|[^\s0-9a-zA-Z#])\S*((&(.*);)|[^\s0-9a-zA-Z])+\S*')
    emoticons = []
    for word in wordBag:
        if nonword.match(word):
            emoticons.append(word)
    return emoticons
    

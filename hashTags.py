import re

def hashTags(post):
    hash = re.compile('^[#@][a-zA-Z0-9]+')
    tags = []
    # There will possibly be more than one hashtag. 
    # Whilst one may be for the topic as a whole, the others are potentially very useful for sentiment analysis
    # How to determine topic? Perhaps some combination of the various methods
    #   Find if one is mentioned in the tweet - this is likely the topic
    #   Find which one has the most mentions in other tweets? More likely to be a topic
    for token in post:
        if hash.match(token):
            tags.append(token)
    return tags
    
def postMentTopic(tags,post):
    topic = ""
    topicCount = 0
    for tag in tags:
        strippedTag = tag.replace('#','')
        searchTag = re.compile(strippedTag,re.IGNORECASE)
        #iterate over tokens in the post and try to match with hashtag with the # removed
        #   choose the hashtag with the most mentions
        tagCount = 0
        for token in post:
            if searchTag.match(token):
                tagCount += 1
        if (tagCount > topicCount):
            topic = tag
            topicCount = tagCount
    #If no hashtags are mentioned, it can't be used as a method of determining the topic
    return topic
import pandas as pd
from tokenizer import Tokenizer
from stopwords import stopwords
from partsOfSpeechTagging import Pos
from counter import Counter



text = """I am already far north of London, and as I walk in the streets of
Petersburgh, I feel a cold northern breeze play upon my cheeks, which
braces my nerves and fills me with delight.  Do you understand this
feeling?  This breeze, which has travelled from the regions towards
which I am advancing, gives me a foretaste of those icy climes.
Inspirited by this wind of promise, my daydreams become more fervent
and vivid.  I try in vain to be persuaded that the pole is the seat of
frost and desolation; it ever presents itself to my imagination as the
region of beauty and delight.  There, Margaret, the sun is for ever
visible, its broad disk just skirting the horizon and diffusing a
perpetual splendour.  There—for with your leave, my sister, I will put
some trust in preceding navigators—there snow and frost are banished;
and, sailing over a calm sea, we may be wafted to a land surpassing in
wonders and in beauty every region hitherto discovered on the habitable
globe.  Its productions and features may be without example, as the
phenomena of the heavenly bodies undoubtedly are in those undiscovered
solitudes.  What may not be expected in a country of eternal light?  I
may there discover the wondrous power which attracts the needle and may
regulate a thousand celestial observations that require only this
voyage to render their seeming eccentricities consistent for ever."""


tokenizer =Tokenizer(text,stopwords)
tokens=tokenizer.word_tokenizer()

# 1.check if tokenizer is working working
# print(tokens) 


pos=Pos()
tokens1=pos.manual_pos_tags(tokens)
# 2. check if pos is working
# print(tokens)
  


word_freq=Counter(tokens1)

# 3. check if counter is working
# print(word_freq)



max_freq = max(word_freq.values())
for word in word_freq.keys():
    word_freq[word] = word_freq[word]/max_freq

# 4. check if word_freq is working
# print(word_freq)



sent_token =tokenizer.sentence_tokenizer()

# 5. check if sentance tokenizer is working 
# print(sent_token)

sent_score = {}
for sent in sent_token:
    for word in sent.split():
        if word in word_freq.keys():
            if sent not in sent_score.keys():
                sent_score[sent] = word_freq[word]
            else:
                sent_score[sent] += word_freq[word]
        # print(word)

# 6 . check if sentence scoring is working 
# print(sent_score)

dataset = pd.DataFrame(list(sent_score.items()),columns=['sentence','score'])
# 7. visualize the scored sentence
# print(dataset)

from heapq import nlargest
def getTopSentences(sent_score, n):
    topSentences=nlargest(n,sent_score,key=sent_score.get)
    return "".join(topSentences)

# 8. printing the final summary
summary = getTopSentences(sent_score,3)
print(summary)




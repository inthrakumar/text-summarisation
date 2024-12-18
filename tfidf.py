import math
import numpy as np
import pandas as pd
df = pd.read_excel('data.xlsx')

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'the']

def euclidean_distance(a):
    distance = 0
    for i in range(len(a)):
        distance += a[i]*a[i]
    return math.sqrt(distance)

def dot_product(a,b):
    num = 0
    for i in range(len(a)):
        num += (a[i]*b[i])
    return num

def TF_IDF(text, num_sentences):
    sentences = []
    sentence = ""

    for i in range(len(text)):
        if text[i] == '.':
            sentences.append(sentence)
            sentence = ""
            continue
        sentence += text[i]
    
    if sentence:
        sentences.append(sentence)

    words = []


    for sentence in sentences:
        for word in sentence.strip(' ').split():
            for sub in word.split(','):
                if sub.lower() not in stopwords:
                    if sub.lower() not in words:
                        words.append(sub.lower())
    
    tdidf_matrix = []
    td = []
    idf = []
    
    for sentence in sentences:
        sent = []
        for i in range(len(words)):
            count = 0
            total = 0
            for word in sentence.strip(' ').split():
                for sub in word.split(','):
                    total += 1
                    if words[i] == sub.lower():
                        count += 1
            sent.append((count / total) if total > 0 else 0)
        td.append(sent)

    for i in range(len(words)):
        count = 0
        for sentence in sentences:
            flag = 0
            for word in sentence.strip(' ').split():
                for sub in word.split(','):
                    if words[i] == sub.lower():
                        if flag == 0:
                            count += 1
                            flag = 1
        idf.append(math.log(1 + len(sentences)/(1+count)))

    for i in range(len(td)):
        row = []
        for j in range(len(td[0])):
            row.append(td[i][j]*idf[j])
        tdidf_matrix.append(row)

    similarity_matrix = []
    
    for i in range(len(sentences)):
        row = []
        for j in range(len(sentences)):
            if i == j:
                row.append(1)
            else:
                row.append(dot_product(tdidf_matrix[i], tdidf_matrix[j])/(euclidean_distance(tdidf_matrix[i])*euclidean_distance(tdidf_matrix[j])+ 1e-10))
        similarity_matrix.append(row)

    similarity_matrix = np.array(similarity_matrix)
    scores = np.ones(len(sentences)) / len(sentences)
    damping_factor = 0.85
    for _ in range(100):
        new_scores = (1 - damping_factor) / len(sentences) + damping_factor * similarity_matrix.T.dot(scores)
        if np.allclose(scores, new_scores, atol=1e-4):
            break
        scores = new_scores

    ranked_sentences = [(score, sentence) for score, sentence in zip(scores, sentences)]
    ranked_sentences = sorted(ranked_sentences, key=lambda x: x[0], reverse=True)

    summary = " ".join([sentence for _, sentence in ranked_sentences[:num_sentences]])
    print(summary)

# TF_IDF("TextRank is an unsupervised graph-based ranking algorithm for NLP tasks. It is similar to PageRank, but it applies to text data. TextRank builds a graph where sentences or words are nodes, and edges represent relationships. By iteratively updating scores, TextRank identifies the most important sentences or words for summarization or keyword extraction.", 3)

text = """I am already far north of London, and as I walk in the streets of Petersburgh, I feel a cold northern breeze play upon my cheeks, which braces my nerves and fills me with delight.  Do you understand this feeling?  This breeze, which has travelled from the regions towards which I am advancing, gives me a foretaste of those icy climes. Inspirited by this wind of promise, my daydreams become more fervent and vivid.  I try in vain to be persuaded that the pole is the seat of frost and desolation; it ever presents itself to my imagination as the region of beauty and delight.  There, Margaret, the sun is for ever visible, its broad disk just skirting the horizon and diffusing a perpetual splendour.  There—for with your leave, my sister, I will put some trust in preceding navigators—there snow and frost are banished; and, sailing over a calm sea, we may be wafted to a land surpassing in wonders and in beauty every region hitherto discovered on the habitable globe.  Its productions and features may be without example, as the phenomena of the heavenly bodies undoubtedly are in those undiscovered solitudes.  What may not be expected in a country of eternal light?  I may there discover the wondrous power which attracts the needle and may regulate a thousand celestial observations that require only this voyage to render their seeming eccentricities consistent for ever."""

text1=df.iloc[0,1]

TF_IDF(text1,4)

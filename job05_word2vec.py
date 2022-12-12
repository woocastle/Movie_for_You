import pandas as pd
from gensim.models import Word2Vec

review_word = pd.read_csv('./crawling_data/one_sentences.csv')
review_word.info() # 임배딩은 차원 줄여주기

one_sentence_reviews = list(review_word['reviews'])
cleaned_tokens = []
for sentence in one_sentence_reviews:
    token = sentence.split()
    cleaned_tokens.append(token)
# 의미차원 안에 좌표를 배치하고 # 차원이 너무 크면 차원에 갇히기 때문에 줄여준다. cleaned_tokens은 좌표를 가지게 한다.
embedding_model = Word2Vec(cleaned_tokens, vector_size=100,
                           window=4, min_count=20, workers=8, epochs=100, sg=1)
# 객체를 만든다. # 의미가 클수록 큰값 의미가 작을수록 작은값 # 4개의 문장으로 보겠다. # 20개 이하는 버리겠다.
# CPU를 몇개 쓸껀지 여긴 8개 쓸꺼다. # 알고리즘이 sg=1 이다. 알고리즘 지칭함

embedding_model.save('./models/word2vec_movie_review.model')
print(list(embedding_model.wv.index_to_key))
print(len(embedding_model.wv.index_to_key))
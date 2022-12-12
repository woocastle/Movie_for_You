# 형태소의 출현빈도 (job06까진 유사 의미)
# TFIDF : T F = 텍스트 프리퀀시(한문장에 몇번나오는지) / I = inverse / D F = 도큐멘트 프리퀀시(문서에서 몇번나오느냐)
# '영화'라는 형태소는 문장이 비슷한지 찾기위해선 안중요. =>'영화'한테는 패널티를 줘야함
# => 역수를 취해서 곱함
# => 문장의 유사도 찾는 것 : '단어의 빈도수'를 볼 것
# (잡6에선 '단어의 유사도'를 보았음)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer     # 문장에 벡터좌표가 만들어짐.
from scipy.io import mmwrite, mmread                            # 그 행렬을 저장할 때
import pickle

#모델만들어서 저장할 것
df_reviews = pd.read_csv('./crawling_data/one_sentences.csv')
df_reviews.info()

tfidf = TfidfVectorizer()                                   # 좌표
tfidf_matrix = tfidf.fit_transform(df_reviews['reviews'])   # 좌표들의 행렬
print(tfidf_matrix[0].shape)                                # 매트릭스의 쉐잎을 보겠다
with open('./models/tfidf.pickle', 'wb') as f:
    pickle.dump(tfidf, f)

mmwrite('./models/tfidf_movie_review.mtx', tfidf_matrix)    # 좌표값을 만들어 주는 애 # 매트릭스이기 때문에 mmwrite로 저장하는 것



# 모델 저장은 됐고 유사한 문장 찾아보겠 => 잡08



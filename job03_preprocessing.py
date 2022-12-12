import pandas as pd
from konlpy.tag import Okt
import re

df = pd.read_csv('./crawling_data/review_final.csv')
df.info()
print(df.head())

df_stopwords = pd.read_csv('./stopwords.csv', index_col=0)  # 불용어
stopwords = list(df_stopwords['stopwords'])     # 겨울왕국 기준으로 추가된
stopwords = stopwords + ['안나', '제니퍼', '미국', '중국', '영화', '감독', '리뷰',
                         '연출', '장면', '주인공', '되어다', '출연', '싶다', '올해', '엘사']
                        # 겨울왕국 기준으로 추가된. Run 안돌려보겠음! 하지만 달라질 것! (안나랑 제니퍼 때문에 추천된 듯한 라이크 크레이지)

# okt = Okt()
# df['clean_reviews'] = None
# for idx, review in enumerate(df.reviews[:10]):
#     review = re.sub('[^가-힣 ]', ' ', review)
#     df.loc[idx, 'clean_reviews'] = review
# print(df.clean_reviews)
#
# token = okt.pos(df.clean_reviews[0], stem=True)        # pos : 특정 품사만 뽑아내려고 # 품사가 뭔지 까지 알려주는 pos. 명사, 동사, 형용사
#     # 부사,, 형용사도 스템트루하면 동사가됨  # 차원이 많아지면 안좋으니 스템트루 주는 것. 형용사는 어떻게 쓰느냐에 따라
#     # 원형이 없는 '특히' 같은 건 많이 없
# print(token)


okt = Okt()
df['clean_reviews'] = None
count = 0
for idx, review in enumerate(df.reviews):
    count += 1
    if count % 10 == 0:     # 10개 마다 점찍고
        print('.', end='')
    if count % 1000 == 0:   # 10줄마다 줄바꿈 (10*100=1000)  # 점 100개찍혀야 한줄내려감
        print()
    review = re.sub('[^가-힣 ]', ' ', review)     # 한글만 남긴
    df.loc[idx, 'clean_reviews'] = review
    token = okt.pos(review, stem=True)      # 형태사 단위로 쪼갠
    df_token = pd.DataFrame(token, columns=['word', 'class'])  # 토큰을 데이터 프레임으로 만들 것
    df_token = df_token[(df_token['class'] == 'Noun') |        # 조건인덱싱 #클래스 컬럼의 값이 명사인 것
                        (df_token['class'] == 'Verb') |
                        (df_token['class'] == 'Adjective')]    # 아름다운=>아름답다와 같은 형용사는 의미 있.
    words = []    # 불용어 제거
    for word in df_token.word:
        if len(word) > 1:
            if word not in list(df_stopwords.stopword):
                words.append(word)
    cleaned_sentence = ' '.join(words) # 형태소들로 되어있으니 하나로 이어붙이겠
    df.loc[idx, 'clean_reviews'] = cleaned_sentence     # 줄여서 클린 리뷰스라는 컬럼으로
    # print(df.clean_reviews)
print(df.head(30))  #30개만 볼
df.dropna(inplace=True)
df.to_csv('./crawling_data/cleaned_reviews_2016_2022.csv', index=False)

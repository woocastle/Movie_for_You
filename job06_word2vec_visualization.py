# 거리 & 시각화
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE   # 차원 축소 알고리즘    # 100차원으로 축소해서 좌표하나당 100개씩
from matplotlib import font_manager, rc
import matplotlib as mpl

font_path = './malgun.ttf'          # 폰트 필요
font_name = font_manager.FontProperties(fname=font_path).get_name()
mpl.rcParams['axes.unicode_minus']=False
rc('font', family=font_name)        # 플랏에 한글 쓰려고 하는 것 안깨지고

embedding_model = Word2Vec.load('./models/word2vec_movie_review.model')
key_word = '사랑'
sim_words = embedding_model.wv.most_similar(key_word, topn=10)    # 모스트시뮬러라는 함수에 키워드 주고 10개만 보겠
print(sim_words)                                                  # 가장 유사한 단어는 키워드와 가장 가까운.

vectors = []
labels = []

for label, _ in sim_words:
    labels.append(label)
    vectors.append(embedding_model.wv[label])
print(vectors[0])
print(len(vectors[0]))

df_vector = pd.DataFrame(vectors)    # 벡터가지고 데이터 프레임 만들기
print(df_vector)
# exit()

# 2차원으로 줄여보겠. 차원을 줄여주는 모델
tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500)           # 방향이나 거리 유지하며 차원축소 하는 것이 => pca
new_value = tsne_model.fit_transform(df_vector)
df_xy = pd.DataFrame({'words':labels, 'x':new_value[:, 0], 'y':new_value[:, 1]})    # 유사단어만 들어가 있는 상태 : 10개
df_xy.loc[len(df_xy)] = (key_word, 0, 0)   # 키워드 좌표도 넣겠 # 0~9번은 10개 차있고, 10번 인덱스(len)에 데이터 추가

print(df_xy)


plt.figure(figsize=(8, 8))
plt.scatter(0, 0, s=500, marker='*')   # 마커는 별, 문자열로 줘야.
plt.scatter(df_xy['x'], df_xy['y'])         #df_xy의 x컬럼의 값이 x고, y컬럼의 값이 y


# 형태소도 쓸 거고 선도 그을 것
# for i in range(10):
for i in range(len(df_xy)):
    # a = df_xy.loc[i, ['x', 'y']]
    a = df_xy.loc[[i, 10]] #i번과 10번 2개의 자료를 뽑는다는
    plt.plot(a.x, a.y, '-D', linewidth=1) #3?
    plt.annotate(df_xy.words[i], xytext=(1, 1), xy=(df_xy.x[i], df_xy.y[i]),
                 textcoords='offset points', ha='right', va='bottom')

plt.show()

# 차원축소해서 서로 해볼 수는 있어
# 같은 사랑이라는 의미차원에서도 서로 반대되는 의미가 생길수도 있다
# 차원축소 = 데이터가 날라가는 것. 방향이나 거리 유지하며 차원축소 하는 것이 => pca
# 시각화보다 의미적으로 유사한게 어떤 것들이 있는지가 더 중요. 추천 : 리뷰가 비슷한 애들을 찾아서 추천해줄 것. 사랑이라는 단어.
# 사랑만 추천하는 게 아니라 '애틋하다'와 같은 비슷한 의미 가지고 있는 것도 추천해주기 위해
# 의미적으로 유사한 단어가 많이 나와서. 사랑이 한번 나왔. 이 영화에서 사랑이라는 의미가 중요? 총싸움이 100번 나오면 => 액션


# 사랑 10 살인 100, 사랑 100, 살인 10 형태소의 출현빈도도 봐야함. => 잡07 (지금까진 의미유사한 것 찾아보는)

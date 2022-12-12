import sys # 파이썬 기본 라이브러리
from PyQt5.QtWidgets import * # *주면 모든파일 import함
from PyQt5 import uic
from PyQt5.QtCore import QStringListModel
import pandas as pd
from scipy.io import mmread
import pickle
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import linear_kernel
from konlpy.tag import Okt
import re


form_window = uic.loadUiType('./movie_recommendation_app.ui')[0] # loadUiType이걸 쓰면 class로 만들어줌

class Exam(QWidget, form_window): # 다중 상속 가능 # QWidget 닫기 최소화 등 버튼 활성화
    def __init__(self):
        super().__init__()
        self.setupUi(self) # setupUi가 많은 것을 설정해줌
        self.tfidf_matrix = mmread('./models/tfidf_movie_review.mtx').tocsr() # 추천 시스템을 쓰기위해 불러옴
        with open('./models/tfidf.pickle', 'rb') as f:
            self.tfidf = pickle.load(f)
        self.embedding_model = Word2Vec.load('./models/word2vec_movie_review.model')

        self.df_reviews = pd.read_csv('./crawling_data/one_sentences.csv')
        self.titles = self.df_reviews['titles']
        self.titles = sorted(self.titles)
        for title in self.titles:
            self.combo_box.addItem(title)


        model = QStringListModel() # 자동완성 기능
        model.setStringList(self.titles) # 여기에 리스트가 들어가면 된다.
        completer = QCompleter()
        completer.setModel(model)
        self.line_edit.setCompleter(completer)

        self.combo_box.currentIndexChanged.connect(self.combobox_slot) # 콤보박스 연결
        self.btn_recommend.clicked.connect(self.btn_slot)


    def recommendation_by_movie_title(self, title):
        movie_idx = self.df_reviews[self.df_reviews['titles'] == title].index[0]  # d..f에서 겨울왕국  # ['titles'] #인덱스 0번째?
        cosin_sim = linear_kernel(self.tfidf_matrix[movie_idx], self.tfidf_matrix)
        # 겨울왕국과 리스트에 있는 모든 영화의 유사도 값.   # 리뷰 4천여개 9만9천여개 tfidf값을 쫙만들었고 문장의 개수만큼 생겼을 것
        recommendation = self.getRecommendation(cosin_sim)
        recommendation = '\n'.join(list(recommendation[1:]))
        self.lbl_recommend.setText(recommendation)



    def recommendation_by_key_word(self, key_word):
        sim_word = self.embedding_model.wv.most_similar(key_word, topn=10)
        words = [key_word]
        for word, _ in sim_word:
            words.append(word)
        print(words)
        sentence = []
        count = 11
        for word in words:
            sentence = sentence + [word] * count
            count -= 1
        sentence = ' '.join(sentence)
        print(sentence)
        sentence_vec = self.tfidf.transform([sentence])
        cosin_sim = linear_kernel(sentence_vec, self.tfidf_matrix)
        recommendation = self.getRecommendation(cosin_sim)
        recommendation = '\n'.join(list(recommendation[:10]))
        self.lbl_recommend.setText(recommendation)


    def recommendation_by_sentence(self, key_word):
        # 얘도 벡터화하고 전처리해야. => 매트릭스 구하고, 코사인 매트릭스
        review = re.sub('[^가-힣 ]', ' ', key_word)
        okt = Okt()
        token = okt.pos(review, stem=True)
        df_token = pd.DataFrame(token, columns=['word', 'class'])
        df_token = df_token[(df_token['class']=='Noun') |
                            (df_token['class']=='Verc') |
                            (df_token['class']=='Adjective')]
        words = []
        for word in df_token.word:                               # 불용어 처리는 생략
            if 1 < len(word):
                words.append(word)
        cleaned_sentence = ' '.join(words)
        print(cleaned_sentence)
        sentence_vec = self.tfidf.transform([cleaned_sentence])       # 센텐스 벡터 만들기
        cosin_sim = linear_kernel(sentence_vec, self.tfidf_matrix)
        recommendation = self.getRecommendation(cosin_sim)
        recommendation = '\n'.join(list(recommendation[:10]))
        self.lbl_recommend.setText(recommendation)


    def btn_slot(self): # title로 함수를 만들어서 묶어 줬다.
        key_word = self.line_edit.text()
        if key_word in self.titles:
            self.recommendation_by_movie_title(key_word)
        elif key_word in list(self.embedding_model.wv.index_to_key):
            self.recommendation_by_key_word(key_word)
        else:
            self.recommendation_by_sentence(key_word)


    def combobox_slot(self):
        title = self.combo_box.currentText()
        self.recommendation_by_movie_title(title)


    def getRecommendation(self, cosin_sim):  # 코사인 유사도 값이 다 들어있음
        simScore = list(enumerate(cosin_sim[-1]))  # 인덱스 붙이고      # enumerate : 소팅하면서 인덱스  # [-1] : 어차피 하나만 들어있어서 0이어도 됨.. 인덱싱을 한번 해줘야해서 한번 해준 것
        simScore = sorted(simScore, key=lambda x: x[1], reverse=True)  # 정렬(sort), 내림차순 되도록 = 큰값 먼저나오게 (reverse)
        simScore = simScore[:11]  # 0번~10번까지 => 가장 유사한 11개. => 왜 11개일까?
        # 이유 : tfidx에 겨울왕국(자신) 있을거고, 둘다 코사인심하면 1번(?)이 자기자신이 됨. 11개뽑고 같은 겨울왕국(자신) 빼고 다른 10개를 추천
        movie_idx = [i[0] for i in simScore]  # 10개 남기고 i로 받아서 영화의 인덱스 뽑아낸 것 # 무비idx에
        recMovieList = self.df_reviews.iloc[movie_idx, 0]  # 0 = 영화 타이틀.
        return recMovieList  # 타이틀 11개 리턴


if __name__ == "__main__": # 나중에 모듈로 써먹기 위해서
    app = QApplication(sys.argv) # 어플이 어플을 동작하게 하는 기능
    mainWindow = Exam() # 객체는 여기서 만들어 진다.
    mainWindow.show() # 화면에 출력해라
    sys.exit(app.exec_()) # 사용자가 한 액션을 처리하는 것 # exec 무한루프 # 윈도우 종료시 exit
    # 클릭 한 후 시그널이 발생하며 받을 슬롯을 지정해준다.
import os
import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
import io

# KoNLPy(OKT)는 선택적 사용: 설치/빌드가 불가한 환경에서는 정규식 기반 토크나이저로 대체
try:
    from konlpy.tag import Okt  # type: ignore
    _okt: Okt | None = Okt()
except Exception:
    _okt = None

# --- 기본 설정 ---
st.set_page_config(layout="wide", page_title="애니메이션 피드백 분석기")

# --- 앱 제목 및 소개 ---
st.title("🌟 애니메이션 피드백 분석기")
st.markdown("사용자 피드백 데이터를 기반으로 **감성 분포**와 **주요 키워드**를 분석하고 시각화합니다.")
st.markdown("---")

# --- 사이드바: 파일 업로드 및 필터 ---
st.sidebar.header("파일 업로드 및 설정")
uploaded_file = st.sidebar.file_uploader(
    "CSV 또는 Excel 파일 업로드", type=["csv", "xlsx"]
)
st.sidebar.markdown(
    "**참고:** 파일을 업로드하지 않으면 로컬의 '@feedback-data.csv' 또는 예시 데이터로 분석을 시작합니다."
)

# --- 데이터 로드 함수 ---
@st.cache_data
def load_data(file_uploader):
    """파일 업로더를 통해 데이터를 로드하거나, 업로드된 파일이 없으면 로컬 CSV 또는 예시 데이터를 생성합니다."""                                     
    if file_uploader:
        try:
            if file_uploader.name.endswith('.csv'):
                df = pd.read_csv(file_uploader)
            else:
                df = pd.read_excel(file_uploader)
            return df
        except Exception as e:
            st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
            return None
    else:
        # 로컬 기본 CSV 경로 확인
        default_path = os.path.join(os.getcwd(), "@feedback-data.csv")
        if os.path.exists(default_path):
            try:
                df = pd.read_csv(default_path)
                st.info("업로드된 파일이 없어 로컬 '@feedback-data.csv'를 사용합니다.")
                return df
            except Exception as e:
                st.warning(f"로컬 CSV를 읽는 중 오류가 발생했습니다: {e}. 예시 데이터를 사용합니다.")
        # 예시 데이터 생성
        example_data = {
            'date': pd.to_datetime(['2024-05-01', '2024-05-02', '2024-05-03', '2024-05-04', '2024-05-05']),
            'product': ['마법소녀 마루코', '로봇 친구 용이', '마법소녀 마루코', '로봇 친구 용이', '마법소녀 마루코'],
            'rating': [5, 4, 1, 2, 5],
            'feedback': [
                "정말 재밌고 감동적이었어요! 캐릭터도 너무 귀엽고 그림체가 아름다워요.",
                "스토리가 좀 느린 것 같지만, 로봇 디자인이 아주 마음에 들어요.",
                "결말이 너무 허무해서 실망했어요. 다음 시즌은 기대하기 어려울 것 같아요.",
                "액션 장면이 훌륭하고 박진감 넘쳐서 좋았어요. 사운드 효과도 최고입니다.",
                "OST가 정말 좋아서 계속 듣고 싶어요. 다시 봐도 너무 좋아요. 꼭 보세요."
            ]
        }
        df = pd.DataFrame(example_data)
        st.info("파일이 업로드되지 않아 예시 데이터를 사용합니다.")
        return df

# --- 텍스트 분석 함수들 ---
stopwords_korean = {
    '이', '그', '저', '것', '수', '때', '곳', '내', '나', '널', '분', '님', '과', '의', '에', '와', '은', '는', '다', '고', '면', '로', '를', '게', '의해',
    '정말', '진짜', '너무', '아주', '안', '못', '또', '잘', '이렇다', '저렇다', '그렇다', '하다', '있다', '없다', '되다', '않다'
}

def _tokenize_korean(text: str) -> list[str]:
    # OKT가 있으면 명사 기준, 없으면 한글 2자 이상 토큰 정규식 사용
    text = re.sub(r"[^가-힣\s]", " ", text)
    if _okt is not None:
        try:
            return [n for n in _okt.nouns(text) if len(n) > 1]
        except Exception:
            pass
    return re.findall(r"[가-힣]{2,}", text)

def analyze_sentiment(text):
    """간단한 키워드 기반 감성 분석"""
    positive_keywords = ['재밌', '감동', '좋', '아름답', '귀엽', '훌륭', '최고', '마음에 들', '최고', '사랑', '추천']
    negative_keywords = ['실망', '아쉽', '허무', '별로', '지루', '느리', '단점']

    tokens = _tokenize_korean(text)
    
    pos_score = sum(1 for word in tokens if any(keyword in word for keyword in positive_keywords))
    neg_score = sum(1 for word in tokens if any(keyword in word for keyword in negative_keywords))
    
    if pos_score > neg_score:
        return '긍정'
    elif neg_score > pos_score:
        return '부정'
    else:
        return '중립'

def extract_keywords(text_series):
    """피드백 텍스트에서 명사 키워드 추출"""
    text_combined = " ".join(text_series.astype(str).tolist())
    nouns = _tokenize_korean(text_combined)
    filtered_nouns = [n for n in nouns if len(n) > 1 and n not in stopwords_korean]
    return Counter(filtered_nouns)

# --- 메인 앱 로직 ---
df = load_data(uploaded_file)

if df is not None:
    st.sidebar.subheader("데이터 필터링")
    
    # 날짜 필터
    df['date'] = pd.to_datetime(df['date'])
    min_date = df['date'].min()
    max_date = df['date'].max()
    date_range = st.sidebar.slider(
        "기간 선택",
        min_value=min_date.date(),
        max_value=max_date.date(),
        value=(min_date.date(), max_date.date())
    )
    df_filtered = df[(df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])]

    # 제품명 필터
    if 'product' in df.columns:
        unique_products = df['product'].unique().tolist()
        selected_products = st.sidebar.multiselect(
            "애니메이션 선택",
            options=unique_products,
            default=unique_products
        )
        df_filtered = df_filtered[df_filtered['product'].isin(selected_products)]

    if st.button("분석 실행"):
        if df_filtered.empty:
            st.warning("선택된 필터에 해당하는 데이터가 없습니다. 필터를 조정해 주세요.")
        else:
            with st.spinner('피드백을 분석 중입니다... 🚀'):
                st.balloons()

                # 감성 분석 실행
                df_filtered['sentiment'] = df_filtered['feedback'].apply(analyze_sentiment)

                # 주요 결과 시각화
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 감성 분포 분석")
                    sentiment_counts = df_filtered['sentiment'].value_counts().reset_index()
                    sentiment_counts.columns = ['sentiment', 'count']
                    fig_sentiment = px.pie(
                        sentiment_counts,
                        names='sentiment',
                        values='count',
                        title='전체 피드백 감성 분포',
                        color='sentiment',
                        color_discrete_map={'긍정': 'lightgreen', '중립': 'yellow', '부정': 'salmon'}
                    )
                    st.plotly_chart(fig_sentiment, use_container_width=True)
                
                with col2:
                    st.subheader("💡 주요 키워드")
                    keywords = extract_keywords(df_filtered['feedback'])
                    if keywords:
                        # 워드 클라우드 생성
                        # OS별 폰트 경로 탐색 (Windows/Streamlit Cloud Linux)
                        font_candidates = [
                            r"C:\\Windows\\Fonts\\malgun.ttf",  # Windows
                            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Debian/Ubuntu
                            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",  # Noto CJK
                        ]
                        font_path = next((p for p in font_candidates if os.path.exists(p)), None)
                        wc = WordCloud(
                            font_path=font_path,
                            width=800,
                            height=400,
                            background_color='white'
                        ).generate_from_frequencies(keywords)
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)
                        
                        # 키워드 테이블
                        st.markdown("---")
                        st.markdown("##### 📌 자주 언급된 키워드 (상위 10개)")
                        top_keywords = keywords.most_common(10)
                        df_keywords = pd.DataFrame(top_keywords, columns=['키워드', '빈도'])
                        st.table(df_keywords)
                    else:
                        st.info("분석할 키워드가 충분하지 않습니다.")
    else:
        st.info("분석 버튼을 눌러주세요.")


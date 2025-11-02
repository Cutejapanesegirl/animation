import os
import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
import io
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
        # 예시 데이터 생성 (실제 애니메이션 이름과 장르 키워드 포함)
        example_data = {
            'date': pd.to_datetime(['2024-04-01', '2024-04-02', '2024-04-03', '2024-04-04', '2024-04-05',
                                    '2024-04-06', '2024-04-07', '2024-04-08', '2024-04-09', '2024-04-10',
                                    '2024-04-11', '2024-04-12']),
            'product': ['Attack on Titan', 'Demon Slayer', 'Your Name', 'Jujutsu Kaisen', 'Haikyuu!!',
                       'Spirited Away', 'One Punch Man', 'Mob Psycho 100', 'Death Note', 'Naruto',
                       'Hunter x Hunter', 'The Promised Neverland'],
            'rating': [4.8, 4.9, 5.0, 4.7, 4.8, 5.0, 4.6, 4.7, 4.9, 4.5, 4.8, 4.7],
            'feedback': [
                "액션, 생존, 진격의 거인 세계관이 정말 흥미로웠어요.",
                "판타지, 액션, 애니메이션 퀄리티가 최고입니다!",
                "판타지, 로맨스, 감동적인 스토리와 OST가 인상적이에요.",
                "액션, 판타지, 주술 배틀 시스템이 신선했어요.",
                "스포츠, 성장, 팀워크가 정말 멋진 작품이에요.",
                "판타지, 모험, 미야자키 하야오 감독의 걸작입니다!",
                "액션, 코미디, 시즈엔이 너무 강해서 웃겨요.",
                "액션, 일상, 초능력과 성장 이야기가 좋아요.",
                "미스터리, 심리, 뇌성 게임이 정말 흥미로웠습니다.",
                "액션, 성장, 닌자의 세계관이 멋졌어요.",
                "판타지, 모험, 헌터 시험과 넨 시스템이 좋아요.",
                "미스터리, 서스펜스, 반전이 인상적이었어요."
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
    """피드백 텍스트에서 애니메이션 장르 키워드 추출"""
    # 애니메이션 장르/내용 키워드 사전
    genre_keywords = {
        '하렘', '후宫', '백합', 'GL', 'BL', '야오이', '야리', '코미디', '개그',
        '판타지', '마법', 'SF', '사이파이', '공포', '호러', '스릴러', '미스터리',
        '액션', '배틀', '전투', '무협', '권선징악', '정의', '레토리카',
        '이세계', '전생', '현대', '중세', '빅토리아', '근대', '미래', '포스트아포칼립스',
        '로맨스', '순애', '러브코미디', '삼각관계', '연애', '카풀', '노래',
        '일상', '라이트', '편안', '힐링', '미소녀', '소년만화',
        '배틀', '경쟁', '토너먼트', '탈출', '생존', '학교', '학원',
        '음악', '밴드', '아이돌', '오케스트라', '발레', '춤',
        '요리', '음식', '식당', '카페', '베이킹',
        '운동', '야구', '축구', '농구', '배구', '테니스', '수영', '승마',
        '게임', 'VR', 'MMORPG', 'MOBA', '카드게임',
        '역사', '다큐', '전기', '신화', '전설'
    }
    
    text_combined = " ".join(text_series.astype(str).tolist())
    nouns = _tokenize_korean(text_combined)
    
    # 장르 키워드만 필터링
    filtered_keywords = []
    for noun in nouns:
        if len(noun) > 1 and noun not in stopwords_korean:
            # 정확히 일치하는 키워드 찾기
            for genre in genre_keywords:
                if genre in noun:
                    filtered_keywords.append(genre)
                    break
    
    return Counter(filtered_keywords)

# --- 추천 알고리즘 함수들 ---
def create_user_animation_matrix(df):
    """사용자-애니메이션 평점 매트릭스 생성"""
    # date를 사용자 ID로 간주 (실제로는 별도 user_id 컬럼이 있어야 함)
    user_ratings = df.groupby(['date', 'product'])['rating'].mean().reset_index()
    user_ratings.columns = ['user_id', 'product', 'rating']
    
    # 피벗 테이블 생성
    matrix = user_ratings.pivot_table(
        index='product', 
        columns='user_id', 
        values='rating', 
        aggfunc='mean'
    ).fillna(0)
    
    return matrix

def get_similar_animations(target_animation, df, top_n=3):
    """특정 애니메이션과 유사한 애니메이션 추천"""
    try:
        matrix = create_user_animation_matrix(df)
        
        if target_animation not in matrix.index:
            return []
        
        # 코사인 유사도 계산 (item-based collaborative filtering)
        target_vector = matrix.loc[target_animation].values.reshape(1, -1)
        similarities = cosine_similarity(target_vector, matrix.values)[0]
        
        similarity_df = pd.DataFrame({
            'animation': matrix.index,
            'similarity': similarities
        })
        
        recommendations = similarity_df[
            similarity_df['animation'] != target_animation
        ].nlargest(top_n, 'similarity')
        
        return recommendations.to_dict('records')
    
    except Exception as e:
        st.warning(f"추천 생성 중 오류 발생: {e}")
        return []

def get_user_based_recommendations(df, target_animations, top_n=3):
    """사용자가 본 애니메이션 기반으로 추천 (사용자 기반 협업 필터링)"""
    try:
        matrix = create_user_animation_matrix(df)
        
        if matrix.empty or len(matrix.index) < 2:
            return []
        
        available_animations = [a for a in target_animations if a in matrix.index]
        if not available_animations:
            return []
        
        user_preferences = matrix.loc[available_animations].mean(axis=0)
        user_vector = user_preferences.values.reshape(1, -1)
        similarities = cosine_similarity(user_vector, matrix.values)[0]
        
        similarity_df = pd.DataFrame({
            'animation': matrix.index,
            'similarity': similarities
        })
        
        recommendations = similarity_df[
            ~similarity_df['animation'].isin(target_animations)
        ].nlargest(top_n, 'similarity')
        
        return recommendations.to_dict('records')
    
    except Exception as e:
        st.warning(f"추천 생성 중 오류 발생: {e}")
        return []

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

                # 주요 결과 시각화
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 평점 분포 분석")
                    # 평점 히스토그램
                    fig_rating = px.histogram(
                        df_filtered,
                        x='rating',
                        nbins=20,
                        title='평점 분포 (1-5점)',
                        labels={'rating': '평점', 'count': '횟수'}
                    )
                    fig_rating.update_layout(
                        xaxis=dict(range=[0, 5]),
                        yaxis_title="평가 횟수"
                    )
                    st.plotly_chart(fig_rating, use_container_width=True)
                    
                    # 통계 정보
                    avg_rating = df_filtered['rating'].mean()
                    st.metric("평균 평점", f"{avg_rating:.2f}점")
                    
                    # 인생작 통계
                    if 'masterpiece' in df_filtered.columns:
                        masterpiece_count = df_filtered['masterpiece'].sum() if df_filtered['masterpiece'].dtype == bool else (df_filtered['masterpiece'] == 'TRUE').sum()
                        st.metric("인생작 지정 횟수", f"{masterpiece_count}회")
                
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
                
                # 추천 섹션 추가
                st.markdown("---")
                st.subheader("🎯 애니메이션 추천")
                st.markdown("본 애니메이션 기반으로 **비슷한 취향의 사람들이 좋아하는** 애니메이션을 추천해드립니다.")
                
                # 추천 모드 선택
                recommendation_mode = st.radio(
                    "추천 방식 선택",
                    ["본 애니메이션들 기반 추천", "특정 애니메이션 유사 추천"],
                    horizontal=True
                )
                
                if recommendation_mode == "본 애니메이션들 기반 추천":
                    # 사용자가 선택한 애니메이션들 기반 추천
                    if selected_products:
                        recommendations = get_user_based_recommendations(
                            df_filtered, 
                            selected_products, 
                            top_n=5
                        )
                        if recommendations:
                            st.success(f"✨ **{', '.join(selected_products)}** 기반 추천 결과")
                            for idx, rec in enumerate(recommendations, 1):
                                similarity_score = rec['similarity']
                                animation_name = rec['animation']
                                st.markdown(f"**{idx}. {animation_name}** (유사도: {similarity_score:.2%})")
                        else:
                            st.info("추천할 애니메이션이 충분하지 않습니다. 더 많은 데이터가 필요합니다.")
                    else:
                        st.info("추천을 위해 최소 1개 이상의 애니메이션을 선택해주세요.")
                
                else:
                    # 특정 애니메이션 유사 추천
                    target_animations_for_rec = st.selectbox(
                        "기준 애니메이션 선택",
                        options=unique_products,
                        help="이 애니메이션과 유사한 애니메이션을 찾아드립니다."
                    )
                    
                    if st.button("유사 애니메이션 찾기"):
                        recommendations = get_similar_animations(
                            target_animations_for_rec,
                            df_filtered,
                            top_n=5
                        )
                        if recommendations:
                            st.success(f"✨ **{target_animations_for_rec}**와(과) 유사한 애니메이션")
                            for idx, rec in enumerate(recommendations, 1):
                                similarity_score = rec['similarity']
                                animation_name = rec['animation']
                                st.markdown(f"**{idx}. {animation_name}** (유사도: {similarity_score:.2%})")
                        else:
                            st.info("유사한 애니메이션을 찾을 수 없습니다. 더 많은 데이터가 필요합니다.")
    else:
        st.info("분석 버튼을 눌러주세요.")


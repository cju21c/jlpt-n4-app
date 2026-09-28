import streamlit as st
import json
import random
import time
from google import genai
from google.genai.errors import ServerError

# 1. 페이지 기본 설정 (모바일 화면 최적화)
st.set_page_config(page_title="JLPT N4 학습 앱", page_icon="🌸", layout="centered")

# 2. API 키 설정 및 클라이언트 생성
import os

if "GEMINI_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "여기에_기본키")

client = genai.Client(api_key=GEMINI_API_KEY)

# 3. 데이터 로드 함수 (캐싱 적용으로 속도 향상)
@st.cache_data
def load_words():
    try:
        with open("words.json", "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        st.error("❌ 'words.json' 파일을 찾을 수 없습니다.")
        return []

# 4. AI 피드백 함수
def get_ai_feedback(word_info, user_sentence):
    word = f"{word_info['kanji']} ({word_info['hiragana']})"
    prompt = f"""
너는 친절하고 능숙한 일본어 선생님이야.
학생이 JLPT N4 단어 '{word}'(뜻: {word_info['meaning']})를 활용하여 아래 일본어 문장을 작성했어.

[학생이 작성한 문장]
{user_sentence}

다음 조건에 맞춰 평가해줘:
1. 문법적/구어적으로 올바른지 검토해줘.
2. 어색하거나 틀린 부분이 있다면 자연스러운 문장으로 수정해줘 (N4 수준에 맞는 친근한 구어체 권장).
3. 수정된 문장의 한국어 뜻을 적어줘.
4. 문법적 포인트나 주의할 점을 2~3줄로 짧고 명확하게 설명해줘.
"""
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model='gemini-3.8-flash',
                contents=prompt,
            )
            return response.text
        except ServerError as e:
            if "503" in str(e) and attempt < max_retries:
                time.sleep(2)
            else:
                return "❌ AI 서버에 연결할 수 없습니다. 나중에 다시 시도해주세요."
        except Exception as e:
            return f"❌ 오류 발생: {e}"

# --- 세션 상태 초기화 ---
if "words_db" not in st.session_state:
    st.session_state.words_db = load_words()
    random.shuffle(st.session_state.words_db)
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "score" not in st.session_state:
    st.session_state.score = 0
if "quiz_finished" not in st.session_state:
    st.session_state.quiz_finished = False

# --- UI 화면 구현 ---
st.title("🌸 JLPT N4 단어 & AI 문장 교정")

if not st.session_state.words_db:
    st.stop()

# 퀴즈 종료 화면
if st.session_state.quiz_finished:
    st.balloons() # 축하 효과
    st.success(f"🎉 모든 단어를 다 풀었습니다! 최종 점수: {st.session_state.score} / {len(st.session_state.words_db)}")
    if st.button("🔄 처음부터 다시 풀기", use_container_width=True):
        random.shuffle(st.session_state.words_db)
        st.session_state.current_idx = 0
        st.session_state.score = 0
        st.session_state.quiz_finished = False
        st.rerun()
else:
    total = len(st.session_state.words_db)
    idx = st.session_state.current_idx
    item = st.session_state.words_db[idx]

    # 진행 상태 표시 바
    st.progress((idx) / total)
    st.caption(f"문제 {idx + 1} / {total}")

    # 단어 카드 출력
    st.subheader(f"단어: {item['hiragana']}")
    st.caption(f"한자: {item['kanji']}")

    # 1. 뜻 맞히기
    user_meaning = st.text_input("1) 한국어 뜻을 입력하세요:", key=f"meaning_{idx}")

    if st.button("정답 확인", key=f"check_btn_{idx}"):
        if user_meaning.strip() == item['meaning']:
            st.success("⭕ 정답입니다!")
            st.session_state.score += 1
        else:
            st.error(f"❌ 아쉽네요! 정답은 '{item['meaning']}' 입니다.")

    st.divider()

    # 2. AI 문장 작성 및 피드백
    st.markdown("### 🤖 AI 일본어 문장 교정")
    st.info("영어나 로마지, 한국어가 섞인 문장으로 작성하셔도 AI가 잘 인식합니다!")
    user_sentence = st.text_area("제시된 단어로 일본어 문장을 만들어보세요:", placeholder="예: Ashita tomo to yakusoku ga arimasu", key=f"sent_{idx}")

    if st.button("✨ AI 선생님에게 문장 검토받기", type="primary", use_container_width=True, key=f"ai_btn_{idx}"):
        if user_sentence.strip():
            with st.spinner("AI 선생님이 문장을 분석하고 있습니다..."):
                feedback = get_ai_feedback(item, user_sentence)
                st.markdown(feedback)
        else:
            st.warning("문장을 입력해 주세요.")

    st.divider()

    # 다음 문제 버튼
    if st.button("다음 문제 ➡️", use_container_width=True):
        if st.session_state.current_idx + 1 < total:
            st.session_state.current_idx += 1
        else:
            st.session_state.quiz_finished = True
        st.rerun()
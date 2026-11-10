import streamlit as st
import json
import random
import time
from google import genai
from google.genai.types import Part
from google.genai.errors import ServerError
import os

# 1. 페이지 기본 설정 (모바일 최적화)
st.set_page_config(page_title="JLPT N4 학습 앱", page_icon="🌸", layout="centered")

# --- 2. API 설정 (Streamlit Secrets 사용) ---
active_api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))

# client 객체 생성 (키가 설정된 경우)
client = genai.Client(api_key=active_api_key) if active_api_key else None


# 3. 데이터 로드 및 저장 함수
def load_words():
    try:
        with open("words.json", "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_words(words_list):
    with open("words.json", "w", encoding="utf-8") as file:
        json.dump(words_list, file, ensure_ascii=False, indent=2)

# 4. 사진에서 단어 자동 추출 (Gemini Vision OCR)
def extract_words_from_image(image_bytes, mime_type):
    if not client:
        st.error("🛑 API Key가 설정되어 있지 않습니다. Streamlit Secrets에 GEMINI_API_KEY를 설정해 주세요.")
        return None

    prompt = """
    이 이미지는 일본어 단어장 페이지야. 이미지 안에 있는 단어 목록을 분석해서 JSON 배열 형식으로만 정밀하게 추출해줘.
    
    응답 조건:
    1. 오직 JSON 배열만 출력할 것.
    2. 각 객체는 "hanja", "hiragana", "meaning" 키를 포함할 것.
    3. 한자가 없는 단어(히라가나만 있는 단어)는 "hanja"에 히라가나 그대로 적을 것.
    
    [JSON 예시]
    [
      {"hanja": "約束", "hiragana": "やくそく", "meaning": "약속"},
      {"hanja": "準備", "hiragana": "じゅんび", "meaning": "준비"}
    ]
    """
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            image_part = Part.from_bytes(data=image_bytes, mime_type=mime_type)
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[image_part, prompt]
            )
            text_res = response.text.strip()
            
            if text_res.startswith("```json"):
                text_res = text_res[7:]
            if text_res.startswith("```"):
                text_res = text_res[3:]
            if text_res.endswith("```"):
                text_res = text_res[:-3]
                
            extracted = json.loads(text_res.strip())
            return extracted
        except Exception as e:
            err_msg = str(e)
            if ("429" in err_msg or "503" in err_msg or "RESOURCE_EXHAUSTED" in err_msg) and attempt < max_retries:
                time.sleep(3)
            else:
                st.error(f"❌ 단어 인식 중 오류가 발생했습니다: {e}")
                return None

# 5. AI 문장 피드백 함수
def get_ai_feedback(word_info, user_sentence):
    if not client:
        return "🛑 API Key가 설정되어 있지 않습니다. Streamlit Secrets에 GEMINI_API_KEY를 설정해 주세요."

    word = f"{word_info.get('hanja', '')} ({word_info.get('hiragana', '')})"
    prompt = f"""
너는 친절하고 능숙한 일본어 선생님이야.
학생이 JLPT N4 단어 '{word}'(뜻: {word_info.get('meaning', '')})를 활용하여 아래 일본어 문장을 작성했어.

[학생이 작성한 문장]
{user_sentence}

다음 조건에 맞춰 평가해줘:
1. 문법적/구어적으로 올바른지 검토해줘.
2. 어색하거나 틀린 부분이 있다면 자연스러운 문장으로 수정해줘 (N4 수준 구어체 권장).
3. 수정된 문장의 한국어 뜻을 적어줘.
4. 문법적 포인트나 주의할 점을 2~3줄로 짧고 명확하게 설명해줘.
"""
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            return response.text
        except Exception as e:
            err_msg = str(e)
            if ("429" in err_msg or "503" in err_msg or "RESOURCE_EXHAUSTED" in err_msg) and attempt < max_retries:
                time.sleep(3)
            else:
                return f"❌ 오류 발생: {e}"

# --- 세션 상태 초기화 (30개 단어 세션 제한 및 오답 노트 관리) ---
def init_quiz_session(custom_words=None):
    if custom_words is not None:
        all_words = custom_words
    else:
        all_words = load_words()
        random.shuffle(all_words)
        all_words = all_words[:30] if len(all_words) > 30 else all_words
        
    st.session_state.words_db = all_words
    st.session_state.current_idx = 0
    st.session_state.score = 0
    st.session_state.quiz_finished = False
    st.session_state.options = []
    st.session_state.answered = False
    st.session_state.user_choice = None
    st.session_state.wrong_words = [] # 이번 세션에서 틀린 단어들을 담을 리스트

if "words_db" not in st.session_state:
    init_quiz_session()

# --- 6. 다재다능한 동적 보기 생성 함수 ---
def generate_options(current_item, all_words, quiz_mode):
    if quiz_mode == "한자 ➡️ 한국어 뜻":
        target_key = "meaning"
    elif quiz_mode == "한자 ➡️ 히라가나 읽기":
        target_key = "hiragana"
    else:  # "히라가나 ➡️ 한자 표기"
        target_key = "hanja"

    correct_answer = current_item.get(target_key, "")
    
    other_candidates = list({w.get(target_key) for w in all_words if w.get(target_key) and w.get(target_key) != correct_answer})
    
    if len(other_candidates) < 3:
        dummy_data = {
            "meaning": ["약속", "준비", "출발", "도착", "공부", "여행"],
            "hiragana": ["やくそく", "じゅんび", "しゅっぱつ", "とうちゃく", "べんきょう", "りょこう"],
            "hanja": ["約束", "準備", "出発", "到着", "勉強", "旅行"]
        }
        for dummy in dummy_data[target_key]:
            if dummy != correct_answer and dummy not in other_candidates:
                other_candidates.append(dummy)

    distractors = random.sample(other_candidates, min(3, len(other_candidates)))
    options = distractors + [correct_answer]
    random.shuffle(options)
    return options, correct_answer


# --- UI 화면 구현 ---
st.title("🌸 JLPT N4 학습 & 단어 스캐너")
st.caption("💡 각 모드별로 30문제씩 풀고, 틀린 단어는 오답 노트로 다시 복습하세요!")

tab1, tab2 = st.tabs(["📝 4지선다 퀴즈", "📸 교재 사진 단어 추가"])

# ================= TAB 1: 4지선다 퀴즈 =================
with tab1:
    quiz_mode = st.radio(
        "🎯 **퀴즈 모드 선택**",
        ["한자 ➡️ 한국어 뜻", "한자 ➡️ 히라가나 읽기", "히라가나 ➡️ 한자 표기"],
        horizontal=True
    )

    if st.session_state.get("last_quiz_mode") != quiz_mode:
        init_quiz_session()
        st.session_state.last_quiz_mode = quiz_mode
        st.rerun()

    if not st.session_state.words_db:
        st.info("등록된 단어가 없습니다. '📸 교재 사진 단어 추가' 탭에서 교재 사진을 올려 단어를 등록해보세요!")
    elif st.session_state.quiz_finished:
        st.balloons()
        st.success(f"🎉 [{quiz_mode}] 퀴즈 완료! 점수: {st.session_state.score} / {len(st.session_state.words_db)}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 새로운 30문제 다시 풀기", use_container_width=True):
                init_quiz_session()
                st.rerun()
        with col2:
            wrong_list = st.session_state.get("wrong_words", [])
            if wrong_list:
                if st.button(f"❌ 틀린 문제만 다시 풀기 ({len(wrong_list)}개)", use_container_width=True, type="primary"):
                    init_quiz_session(custom_words=wrong_list)
                    st.rerun()
            else:
                st.info("✨ 틀린 문제가 없습니다! 완벽해요!")
    else:
        total = len(st.session_state.words_db)
        idx = st.session_state.current_idx
        item = st.session_state.words_db[idx]

        if not st.session_state.options:
            st.session_state.options, st.session_state.correct_answer = generate_options(item, load_words(), quiz_mode)
            st.session_state.answered = False

        st.progress((idx) / total)
        st.caption(f"문제 {idx + 1} / {total}")

        st.markdown("---")
        if quiz_mode == "한자 ➡️ 한국어 뜻":
            st.subheader(f"한자: {item.get('hanja', '')}")
            st.caption(f"발음: {item.get('hiragana', '')}")
            st.markdown("### 1) 알맞은 **한국어 뜻**을 선택하세요:")
        elif quiz_mode == "한자 ➡️ 히라가나 읽기":
            st.subheader(f"한자: {item.get('hanja', '')}")
            st.caption(f"뜻: {item.get('meaning', '')}")
            st.markdown("### 1) 알맞은 **히라가나 읽기**를 선택하세요:")
        else:  
            st.subheader(f"히라가나: {item.get('hiragana', '')}")
            st.caption(f"뜻: {item.get('meaning', '')}")  
            st.markdown("### 1) 알맞은 **한자 표기**를 선택하세요:")

        user_choice = st.radio(
            "보기를 선택하세요",
            st.session_state.options,
            key=f"radio_{idx}_{quiz_mode}",
            label_visibility="collapsed"
        )

        if st.button("정답 확인", key=f"check_btn_{idx}", use_container_width=True):
            st.session_state.answered = True
            st.session_state.user_choice = user_choice
            if user_choice == st.session_state.correct_answer:
                st.session_state.score += 1
            else:
                # 틀린 경우 오답 리스트에 추가 (중복 방지)
                if item not in st.session_state.wrong_words:
                    st.session_state.wrong_words.append(item)

        if st.session_state.answered:
            if st.session_state.user_choice == st.session_state.correct_answer:
                st.success("⭕ 정답입니다!")
            else:
                st.error(f"❌ 아쉽네요! 정답은 '{st.session_state.correct_answer}' 입니다.")

            if st.button("다음 문제 ➡️", key=f"next_btn_{idx}", use_container_width=True, type="primary"):
                if st.session_state.current_idx + 1 < total:
                    st.session_state.current_idx += 1
                    st.session_state.options = []
                    st.session_state.answered = False
                else:
                    st.session_state.quiz_finished = True
                st.rerun()

        st.divider()

        # 2. AI 문장 교정
        st.markdown("### 🤖 AI 일본어 문장 교정 (선택사항)")
        user_sentence = st.text_area("제시된 단어로 일본어 문장을 만들어보세요:", placeholder="예: Ashita tomo to yakusoku ga arimasu", key=f"sent_{idx}")

        if st.button("✨ AI 선생님에게 문장 검토받기", type="primary", use_container_width=True, key=f"ai_btn_{idx}"):
            if user_sentence.strip():
                with st.spinner("AI 선생님이 분석하고 있습니다..."):
                    feedback = get_ai_feedback(item, user_sentence)
                    st.markdown(feedback)
            else:
                st.warning("문장을 입력해 주세요.")

# ================= TAB 2: 사진 스캔 단어 추가 =================
with tab2:
    st.subheader("📸 단어장 사진을 올려 자동으로 등록하세요")
    st.caption("스마트폰 카메라로 교재 페이지를 직접 촬영하거나 갤러리의 이미지 파일을 선택하세요.")

    upload_method = st.radio("업로드 방식 선택", ["📁 갤러리 이미지 선택", "📷 카메라 직접 촬영"], horizontal=True)

    uploaded_file = None
    if upload_method == "📁 갤러리 이미지 선택":
        uploaded_file = st.file_uploader("단어장 사진 업로드 (JPG, PNG)", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("단어장 사진 찍기")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="업로드한 이미지", use_container_width=True)
        
        if st.button("✨ AI 단어 스캔 및 DB 추가 시작", type="primary", use_container_width=True):
            with st.spinner("Gemini AI가 교재의 단어 목록을 읽고 인식하는 중입니다..."):
                bytes_data = uploaded_file.getvalue()
                mime_type = uploaded_file.type if uploaded_file.type else "image/jpeg"
                
                new_words = extract_words_from_image(bytes_data, mime_type)
                
                if new_words and isinstance(new_words, list):
                    current_db = load_words()
                    existing_kanjis = {w.get('hanja') for w in current_db}
                    
                    added_count = 0
                    max_id = max([w.get('id', 0) for w in current_db], default=0)
                    
                    for item in new_words:
                        hanja_val = item.get('hanja')
                        if hanja_val and hanja_val not in existing_kanjis:
                            max_id += 1
                            current_db.append({
                                "id": max_id,
                                "hanja": hanja_val,
                                "hiragana": item.get('hiragana', hanja_val),
                                "meaning": item.get('meaning', '')
                            })
                            added_count += 1
                    
                    save_words(current_db)
                    st.success(f"🎉 성공! 총 {added_count}개의 새로운 단어가 DB에 자동 추가되었습니다!")
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

# --- 2. 사이드바: 개인 API 키 입력 설정 ---
st.sidebar.title("⚙️ 설정")
st.sidebar.markdown("Google AI Studio에서 무료로 발급받은 본인 API 키를 입력하면 개인 쿼터(일일 무료 한도)로 사용할 수 있습니다.")

# 기본 세크릿 키 또는 개인 입력 키 선택
default_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
user_api_key = st.sidebar.text_input("Gemini API Key 입력", value="", type="password", placeholder="AI Studio 키를 여기에 붙여넣으세요")

# 사용자가 입력한 키가 있으면 우선 사용, 없으면 기본 키 사용
active_api_key = user_api_key.strip() if user_api_key.strip() else default_key

if not active_api_key:
    st.sidebar.warning("⚠️ API Key가 설정되지 않았습니다. AI 문장 검토 및 사진 스캔 기능을 위해 키를 입력해 주세요.")

# client 객체 생성 (키가 설정된 경우)
client = genai.Client(api_key=active_api_key) if active_api_key else None

st.sidebar.markdown("---")
st.sidebar.markdown("[👉 Google AI Studio에서 무료 키 발급받기](https://aistudio.google.com/)")


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
        st.error("🛑 API Key가 설정되어 있지 않습니다. 사이드바에서 Gemini API Key를 입력해 주세요.")
        return None

    prompt = """
    이 이미지는 일본어 단어장 페이지야. 이미지 안에 있는 단어 목록을 분석해서 JSON 배열 형식으로만 정밀하게 추출해줘.
    
    응답 조건:
    1. 오직 JSON 배열만 출력할 것.
    2. 각 객체는 "kanji", "hiragana", "meaning" 키를 포함할 것.
    3. 한자가 없는 단어(히라가나만 있는 단어)는 "kanji"에 히라가나 그대로 적을 것.
    
    [JSON 예시]
    [
      {"kanji": "約束", "hiragana": "やくそく", "meaning": "약속"},
      {"kanji": "準備", "hiragana": "じゅんび", "meaning": "준비"}
    ]
    """
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            image_part = Part.from_bytes(data=image_bytes, mime_type=mime_type)
            response = client.models.generate_content(
                model='gemini-3.8-flash',
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
                if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
                    st.error(
                        "🛑 **AI 일일 무료 한도에 도달했습니다!**\n\n"
                        "- 사이드바에 개인 API Key를 입력하여 사용하시거나, 내일 다시 시도해 주세요.\n"
                        "- **📝 4지선다 단어 퀴즈는 제한 없이 계속 이용 가능합니다!** 🎯"
                    )
                else:
                    st.error(f"❌ 단어 인식 중 오류가 발생했습니다: {e}")
                return None

# 5. AI 문장 피드백 함수
def get_ai_feedback(word_info, user_sentence):
    if not client:
        return "🛑 API Key가 설정되어 있지 않습니다. 사이드바에서 Gemini API Key를 입력해 주세요."

    word = f"{word_info['kanji']} ({word_info['hiragana']})"
    prompt = f"""
너는 친절하고 능숙한 일본어 선생님이야.
학생이 JLPT N4 단어 '{word}'(뜻: {word_info['meaning']})를 활용하여 아래 일본어 문장을 작성했어.

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
                model='gemini-3.8-flash',
                contents=prompt,
            )
            return response.text
        except Exception as e:
            err_msg = str(e)
            if ("429" in err_msg or "503" in err_msg or "RESOURCE_EXHAUSTED" in err_msg) and attempt < max_retries:
                time.sleep(3)
            else:
                if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
                    return (
                        "🛑 **AI 일일 이용 한도에 도달했습니다!**\n\n"
                        "• **문장 검토 및 사진 스캔**: 사이드바에 본인의 무료 API Key를 입력하시면 즉시 제한 없이 이용하실 수 있습니다.\n"
                        "• **4지선다 단어 퀴즈**: API를 사용하지 않으므로 제한 없이 언제든 푸실 수 있습니다! 🎯"
                    )
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
if "options" not in st.session_state:
    st.session_state.options = []
if "answered" not in st.session_state:
    st.session_state.answered = False
if "user_choice" not in st.session_state:
    st.session_state.user_choice = None

# --- 6. 다재다능한 동적 보기 생성 함수 ---
def generate_options(current_item, all_words, quiz_mode):
    # 모드에 따른 target_key(정답 field) 설정
    if quiz_mode == "한자 ➡️ 한국어 뜻":
        target_key = "meaning"
    elif quiz_mode == "한자 ➡️ 히라가나 읽기":
        target_key = "hiragana"
    else:  # "히라가나 ➡️ 한자 표기"
        target_key = "kanji"

    correct_answer = current_item[target_key]
    
    # 중복 오답 보기 추출 (정답과 같은 항목 제외)
    other_candidates = list({w[target_key] for w in all_words if w[target_key] != correct_answer and w[target_key]})
    
    # 오답 후보가 부족할 경우 더미 데이터 보충
    if len(other_candidates) < 3:
        dummy_data = {
            "meaning": ["약속", "준비", "출발", "도착", "공부", "여행"],
            "hiragana": ["やくそく", "じゅんび", "しゅっぱつ", "とうちゃく", "べんきょう", "りょこう"],
            "kanji": ["約束", "準備", "出発", "到着", "勉強", "旅行"]
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
st.caption("💡 단어 퀴즈는 무제한 이용 가능하며, AI 기능 공유 한도 초과 시 사이드바에 개인 API Key를 입력해 사용하세요.")

tab1, tab2 = st.tabs(["📝 4지선다 퀴즈", "📸 교재 사진 단어 추가"])

# ================= TAB 1: 4지선다 퀴즈 =================
with tab1:
    if not st.session_state.words_db:
        st.info("등록된 단어가 없습니다. '📸 교재 사진 단어 추가' 탭에서 교재 사진을 올려 단어를 등록해보세요!")
    elif st.session_state.quiz_finished:
        st.balloons()
        st.success(f"🎉 모든 단어를 다 풀었습니다! 최종 점수: {st.session_state.score} / {len(st.session_state.words_db)}")
        if st.button("🔄 처음부터 다시 풀기", use_container_width=True):
            st.session_state.words_db = load_words()
            random.shuffle(st.session_state.words_db)
            st.session_state.current_idx = 0
            st.session_state.score = 0
            st.session_state.quiz_finished = False
            st.session_state.answered = False
            st.session_state.options = []
            st.rerun()
    else:
        # 퀴즈 모드 선택 라디오 버튼
        quiz_mode = st.radio(
            "🎯 **퀴즈 모드 선택**",
            ["한자 ➡️ 한국어 뜻", "한자 ➡️ 히라가나 읽기", "히라가나 ➡️ 한자 표기"],
            horizontal=True
        )

        total = len(st.session_state.words_db)
        idx = st.session_state.current_idx
        item = st.session_state.words_db[idx]

        # 퀴즈 모드가 변경되거나 첫 문제일 때 보기 재생성
        if not st.session_state.options or st.session_state.get("last_quiz_mode") != quiz_mode:
            st.session_state.options, st.session_state.correct_answer = generate_options(item, st.session_state.words_db, quiz_mode)
            st.session_state.last_quiz_mode = quiz_mode
            st.session_state.answered = False

        st.progress((idx) / total)
        st.caption(f"문제 {idx + 1} / {total} (총 {total}단어 보유 중)")

        # 모드별 제시 문제 표시
        st.markdown("---")
        if quiz_mode == "한자 ➡️ 한국어 뜻":
            st.subheader(f"한자: {item['kanji']}")
            st.caption(f"발음: {item['hiragana']}")
            st.markdown("### 1) 알맞은 **한국어 뜻**을 선택하세요:")
        elif quiz_mode == "한자 ➡️ 히라가나 읽기":
            st.subheader(f"한자: {item['kanji']}")
            st.caption(f"뜻: {item['meaning']}")
            st.markdown("### 1) 알맞은 **히라가나 읽기**를 선택하세요:")
        else:  # 히라가나 ➡️ 한자 표기
            st.subheader(f"히라가나: {item['hiragana']}")
            st.caption(f"뜻: {item['meaning']}")  # 동음이의어 구분을 위해 뜻을 힌트로 표시
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

        if st.session_state.answered:
            if st.session_state.user_choice == st.session_state.correct_answer:
                st.success("⭕ 정답입니다!")
            else:
                st.error(f"❌ 아쉽네요! 정답은 '{st.session_state.correct_answer}' 입니다.")

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

        st.divider()

        if st.button("다음 문제 ➡️", use_container_width=True):
            if st.session_state.current_idx + 1 < total:
                st.session_state.current_idx += 1
                st.session_state.options = []
                st.session_state.answered = False
            else:
                st.session_state.quiz_finished = True
            st.rerun()

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
                    existing_kanjis = {w['kanji'] for w in current_db}
                    
                    added_count = 0
                    max_id = max([w.get('id', 0) for w in current_db], default=0)
                    
                    for item in new_words:
                        if item.get('kanji') and item['kanji'] not in existing_kanjis:
                            max_id += 1
                            current_db.append({
                                "id": max_id,
                                "kanji": item['kanji'],
                                "hiragana": item.get('hiragana', item['kanji']),
                                "meaning": item.get('meaning', '')
                            })
                            added_count += 1
                    
                    save_words(current_db)
                    st.session_state.words_db = current_db
                    
                    st.success(f"🎉 성공! 총 {added_count}개의 새로운 단어가 DB에 자동 추가되었습니다!")
                    st.json(new_words)
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

# 2. API 키 설정 및 클라이언트 생성
if "GEMINI_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "여기에_기본키")

client = genai.Client(api_key=GEMINI_API_KEY)

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
    prompt = """
    이 이미지는 일본어 단어장 페이지야. 이미지 안에 있는 단어 목록을 분석해서 JSON 배열 형식으로만 정밀하게 추출해줘.
    
    응답 조건:
    1. 오직 JSON 배열만 출력할 것 (마크다운 ```json 태그 없이 pure JSON만 출력하거나, 추출된 배열만 출력).
    2. 각 객체는 "kanji", "hiragana", "meaning" 키를 포함할 것.
    3. 한자가 없는 단어(히라가나만 있는 단어)는 "kanji"에 히라가나 그대로 적을 것.
    
    [JSON 예시]
    [
      {"kanji": "約束", "hiragana": "やくそく", "meaning": "약속"},
      {"kanji": "準備", "hiragana": "じゅんび", "meaning": "준비"}
    ]
    """
    try:
        image_part = Part.from_bytes(data=image_bytes, mime_type=mime_type)
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=[image_part, prompt]
        )
        text_res = response.text.strip()
        
        # 마크다운 래핑 제거 처리
        if text_res.startswith("```json"):
            text_res = text_res[7:]
        if text_res.startswith("```"):
            text_res = text_res[3:]
        if text_res.endswith("```"):
            text_res = text_res[:-3]
            
        extracted = json.loads(text_res.strip())
        return extracted
    except Exception as e:
        st.error(f"❌ 단어 인식 중 오류가 발생했습니다: {e}")
        return None

# 5. AI 문장 피드백 함수
def get_ai_feedback(word_info, user_sentence):
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
st.title("🌸 JLPT N4 학습 & 단어 스캐너")

# 사이드바 / 탭을 활용한 모드 선택
tab1, tab2 = st.tabs(["📝 퀴즈 풀기", "📸 교재 사진 단어 추가"])

# ================= TAB 1: 퀴즈 풀기 =================
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
            st.rerun()
    else:
        total = len(st.session_state.words_db)
        idx = st.session_state.current_idx
        item = st.session_state.words_db[idx]

        st.progress((idx) / total)
        st.caption(f"문제 {idx + 1} / {total} (총 {total}단어 보유 중)")

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

        # 2. AI 문장 교정
        st.markdown("### 🤖 AI 일본어 문장 교정")
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
            else:
                st.session_state.quiz_finished = True
            st.rerun()

# ================= TAB 2: 사진 스캔 단어 추가 =================
with tab2:
    st.subheader("📸 단어장 사진을 올려 자동으로 등록하세요")
    st.caption("스마트폰 카메라로 교재 페이지를 직접 촬영하거나 갤러리의 이미지 파일을 선택하세요.")

    # 카메라 촬영 및 파일 업로드 2가지 방식 제공
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
                        # 중복 단어 방지 및 ID 자동 부여
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
                    st.json(new_words) # 추출 결과 프리뷰 출력
                else:
                    st.warning("단어를 인식하지 못했습니다. 단어가 명확히 보이도록 다시 찍어주세요.")
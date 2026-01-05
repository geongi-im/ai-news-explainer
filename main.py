import json
import os
import requests
import base64
import time
from bs4 import BeautifulSoup
from datetime import datetime
from dotenv import load_dotenv
from fake_useragent import UserAgent
from google import genai
from google.genai import types
from utils.api_util import ApiUtil, ApiError
from utils.logger_util import LoggerUtil
from utils.telegram_util import TelegramUtil


def getFirstArticleUrl(press_code, date, headers):
    """신문 1면 페이지에서 첫 번째 기사 링크를 추출"""
    url = f"https://media.naver.com/press/{press_code}/newspaper?date={date}"
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        raise Exception(f"🛑 기사 목록 요청 실패: {resp.status_code}")

    soup = BeautifulSoup(resp.text, "html.parser")
    brick_div = soup.find("div", class_="newspaper_brick_item _start_page")
    ul = brick_div.find("ul", class_="newspaper_article_lst") if brick_div else None
    first_li = ul.find("li") if ul else None
    link_tag = first_li.find("a") if first_li else None

    if not (link_tag and link_tag.has_attr("href")):
        raise Exception("🛑 기사 링크를 찾을 수 없습니다.")

    return link_tag["href"]


def getArticleContent(article_url, headers):
    """기사 페이지에서 제목과 본문을 추출"""
    resp = requests.get(article_url, headers=headers)
    if resp.status_code != 200:
        raise Exception(f"🛑 기사 페이지 요청 실패: {resp.status_code}")

    soup = BeautifulSoup(resp.text, "html.parser")

    # 제목
    title_div = soup.find("div", class_="media_end_head_title")
    title = title_div.get_text(strip=True) if title_div else "❌ 제목 없음"

    # 본문
    body_div = soup.find("div", id="newsct_article")
    body = body_div.get_text(separator="\n", strip=True) if body_div else "❌ 본문 없음"

    return title, body


def getGeminiResponse(title, body, max_retries=3, retry_delay=5, success_delay=3):
    """Gemini API를 이용해서 어린이들이 알기쉽게 뉴스 제목과 본문을 분석해서 요약"""
    logger = LoggerUtil().get_logger()
    
    client = genai.Client(
        api_key=os.environ.get("GOOGLE_API_KEY"),
    )

    model = os.environ.get("GEMINI_MODEL")

    system_prompt = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt.md"), "r", encoding="utf-8").read()
    user_prompt = f"""
    뉴스 제목: {title}
    뉴스 본문: {body}
    """

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=user_prompt),
            ],
        ),
    ]
    
    generate_content_config = types.GenerateContentConfig(
        temperature=0.7,
        response_mime_type="application/json",
        system_instruction=[
            types.Part.from_text(text=system_prompt),
        ],
    )

    # 재시도 로직
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_content_config,
            )
            result = response.text
            # 성공 시 3초 대기 (API 서버 부하 방지)
            time.sleep(success_delay)
            logger.info("✅ Gemini API 호출 성공")
            return result
        except Exception as e:
            error_message = str(e)
            # 503(UNAVAILABLE) 및 429(RESOURCE_EXHAUSTED) 모두 처리
            if any(err in error_message for err in ["503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"]):
                wait_time = retry_delay * (2 ** attempt)  # 지수 백오프(5, 10, 20초...)
                logger.warning(f"API 서버 과부하. {wait_time}초 후 재시도... (시도 {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:  # 마지막 시도가 아닌 경우에만 대기
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"API 호출 최대 재시도 횟수 초과: {error_message}")
                    return None
            else:
                logger.error(f"API 호출 중 예상치 못한 오류 발생: {error_message}")
                return None
    
    return None

# AI로 부터 전달받은 정보를 HTML 태그로 변환
def convertToHtml(response, article_url):
    # response를 json 형식으로 변환
    json_response = json.loads(response)
    
    # json에 모든 키가 존재하는지 체크하기
    required_keys = ["title", "summary", "meaning", "importance", "impact_on_us", "food_for_thought", "key_terms"]
    for key in required_keys:
        if key not in json_response:
            raise Exception(f"🛑 {key} 키가 존재하지 않습니다.")
    
    # key_terms를 HTML 리스트로 변환
    def generate_key_terms_html(key_terms):
        terms_html = []
        for term_info in key_terms:
            term_html = f"""
                        <li class="border-l-4 border-green-500 pl-4">
                            <strong class="font-bold text-green-700 text-lg block">
                                {term_info["term"]}
                            </strong>
                            <span class="text-gray-600">
                                {term_info["definition"]}
                            </span>
                        </li>"""
            terms_html.append(term_html)
        return "".join(terms_html)
    
    # HTML 템플릿 생성
    html_response = f"""
    <div class="max-w-3xl mx-auto my-8 sm:my-12">
        <div class="bg-white rounded-xl shadow-lg overflow-hidden">
            <div class="p-6 sm:p-10">
                
                <h1 class="text-3xl sm:text-4xl font-bold text-gray-900 leading-tight mb-4">
                    {json_response["title"]}
                </h1>
                
                <div class="bg-blue-50 border-l-4 border-blue-400 p-4 rounded-r-lg mb-10">
                    <p class="text-base sm:text-lg font-semibold text-blue-800">
                        <span class="font-bold">한 줄 요약!</span>
                        {json_response["summary"]}
                    </p>
                </div>

                <div class="space-y-10">
                    <div>
                        <h2 class="text-2xl font-bold text-gray-800 mb-3 flex items-center">
                            <span class="text-3xl mr-3">💡</span>
                            무슨 뜻일까요?
                        </h2>
                        <p class="text-gray-700 text-base sm:text-lg leading-relaxed">
                            {json_response["meaning"]}
                        </p>
                    </div>

                    <div>
                        <h2 class="text-2xl font-bold text-gray-800 mb-3 flex items-center">
                            <span class="text-3xl mr-3">🌍</span>
                            이게 왜 중요할까요?
                        </h2>
                        <p class="text-gray-700 text-base sm:text-lg leading-relaxed">
                            {json_response["importance"]}
                        </p>
                    </div>

                    <div>
                        <h2 class="text-2xl font-bold text-gray-800 mb-3 flex items-center">
                            <span class="text-3xl mr-3">👨‍👩‍👧‍👦</span>
                            나와 우리 가족에게는?
                        </h2>
                        <p class="text-gray-700 text-base sm:text-lg leading-relaxed">
                            {json_response["impact_on_us"]}
                        </p>
                    </div>

                    <div class="bg-yellow-50 border-2 border-dashed border-yellow-400 p-6 rounded-lg">
                        <h2 class="text-xl font-bold text-yellow-900 mb-3 flex items-center">
                            <span class="text-2xl mr-3">🤔</span>
                            슬기롭게 생각해 보기
                        </h2>
                        <p class="text-yellow-800 text-base sm:text-lg leading-relaxed">
                            {json_response["food_for_thought"]}
                        </p>
                    </div>

                </div>

                <hr class="my-10 border-gray-200">

                <div class="bg-gray-50 p-6 rounded-lg">
                    <h2 class="text-2xl font-bold text-gray-800 mb-5 flex items-center">
                        <span class="text-3xl mr-3">🔍</span>
                        오늘의 경제 용어 돋보기
                    </h2>
                    <ul class="space-y-4">
                        {generate_key_terms_html(json_response["key_terms"])}
                    </ul>
                </div>

                <hr class="my-8 border-gray-200">

                <div class="text-center">
                    <a href="{article_url}" 
                       target="_blank" 
                       rel="noopener noreferrer"
                       class="inline-flex items-center px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition-colors duration-200 shadow-md hover:shadow-lg">
                        <span class="text-xl mr-2">📰</span>
                        뉴스 원본 보기
                        <span class="ml-2">↗</span>
                    </a>
                </div>

            </div>
        </div>
    </div>
    """
    return html_response
    
    

if __name__ == "__main__":
    logger = LoggerUtil().get_logger()

    # 환경변수 로드
    load_dotenv()

    # 필수 환경변수 체크
    required_env_vars = [
        "PRESS_CODE",
        "GOOGLE_API_KEY",
        "GEMINI_MODEL",
        "BASE_URL",
        "TELEGRAM_CHAT_TEST_ID",
        "TELEGRAM_CHAT_ID",
        "TELEGRAM_BOT_TOKEN"
    ]

    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        error_message = f"🛑 필수 환경변수가 설정되지 않았습니다: {', '.join(missing_vars)}"
        logger.error(error_message)
        raise ValueError(error_message)

    # 환경변수 체크 완료 후 유틸리티 초기화
    api_util = ApiUtil()
    telegram = TelegramUtil()
    press_code = os.getenv("PRESS_CODE")

    # 2. 날짜 및 헤더 설정
    today = datetime.today().strftime("%Y-%m-%d")
    today_str = datetime.today().strftime("%Y%m%d")
    ua = UserAgent()
    headers = {"User-Agent": ua.random}

    # 3. 기사 링크 크롤링 → 내용 크롤링
    try:
        article_url = getFirstArticleUrl(press_code, today_str, headers)
        logger.info(f"✅ 기사 링크: {article_url}")

        title, body = getArticleContent(article_url, headers)

        logger.info("\n📰 [기사 제목] \n" + title)
        logger.info("\n📝 [본문 내용] \n" + body)

        response = getGeminiResponse(title, body)
        if response is None:
            logger.error("❌ Gemini API 호출 실패")
            raise Exception("Gemini API 호출 실패")
        
        logger.info("\n📝 [Gemini 응답] \n" + response)

        html_response = convertToHtml(response, article_url)
        logger.info("\n📝 [HTML 응답] \n" + html_response)

        # API 포스트 생성
        try:
            logger.info("API 포스트 생성 시작")
            api_util.create_post(
                title=f"{today} 오늘의 어린이 뉴스",
                content=html_response,
                category="어린이뉴스",
                writer="admin",
                image_paths=[],
                thumbnail_image_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img', 'main.png')
            )
            logger.info("API 포스트 생성 완료")
        except ApiError as e:
            error_message = f"❌ [ai-news-explainer] API 오류 발생\n\n{e.message}"
            telegram.send_test_message(error_message)
            logger.error(f"API 포스트 생성 오류: {e.message}")


    except Exception as e:
        logger.error(f"❌ 에러 발생: {e}")

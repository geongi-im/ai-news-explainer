import requests
from typing import List, Optional
import os
from PIL import Image
import io
from utils.logger_util import LoggerUtil

class ApiError(Exception):
    """API 호출 관련 커스텀 예외"""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API Error (Status: {status_code}): {message}")

class ApiUtil:
    def __init__(self):
        self.base_url = "http://localhost/api"
        self.headers = {
            "Accept": "application/json"
        }
        self.max_file_size = 1 * 1024 * 1024  # 1MB
        self.max_width = 800  # 최대 너비
        self.logger = LoggerUtil().get_logger()

    def _compress_image(self, image_path: str):
        """이미지 압축"""
        try:
            with Image.open(image_path) as img:
                # 이미지 크기 조정
                if img.width > self.max_width:
                    ratio = self.max_width / img.width
                    new_height = int(img.height * ratio)
                    img = img.resize((self.max_width, new_height), Image.Resampling.LANCZOS)
                
                # 이미지 품질 조정
                buffer = io.BytesIO()
                format = img.format if img.format else 'PNG'
                
                if format == 'PNG':
                    img.save(buffer, format=format, optimize=True)
                else:
                    img.save(buffer, format=format, quality=85, optimize=True)
                
                compressed_image = buffer.getvalue()
                
                # 압축 후에도 크기가 큰 경우 추가 압축
                quality = 85
                while len(compressed_image) > self.max_file_size and quality > 30:
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG', quality=quality, optimize=True)
                    compressed_image = buffer.getvalue()
                    quality -= 10
                
                self.logger.info(f"이미지 압축 완료: {image_path} (크기: {len(compressed_image)/1024:.1f}KB)")
                return compressed_image, format.lower()
        except Exception as e:
            self.logger.error(f"이미지 압축 실패: {image_path} - {str(e)}")
            raise

    def create_post(self, title: str, content: str, category: str, writer: str, image_paths: Optional[List[str]] = None):
        """게시글 생성 API 호출"""
        url = f"{self.base_url}/board-content"
        
        try:
            if image_paths:
                self.logger.info(f"게시글 생성 시작 (이미지 포함) - 제목: {title}")
                # 이미지와 함께 게시글 등록
                files = {}
                for i, image_path in enumerate(image_paths):
                    if os.path.exists(image_path):
                        try:
                            compressed_image, format = self._compress_image(image_path)
                            # 원본 파일명 사용
                            original_filename = os.path.basename(image_path)
                            # 각 이미지를 배열로 전송
                            files[f'image[{i}]'] = (original_filename, compressed_image, f'image/{format}')
                            self.logger.debug(f"이미지 {i+1} 추가: {original_filename}")
                        except Exception as e:
                            self.logger.error(f"이미지 처리 실패: {image_path} - {str(e)}")
                            continue
                
                if not files:
                    error_msg = "처리 가능한 이미지가 없습니다."
                    self.logger.error(error_msg)
                    raise ApiError(400, error_msg)
                
                data = {
                    "title": title,
                    "content": content,
                    "category": category,
                    "writer": writer
                }
                
                try:
                    # 요청 데이터 로깅 추가
                    self.logger.debug(f"API 요청 데이터: {data}")
                    self.logger.debug(f"파일 데이터: {[f'{k}: {v[0]}' for k, v in files.items()]}")
                    
                    # multipart/form-data로 전송 시에는 Content-Type 헤더를 제거 (requests가 자동으로 설정)
                    headers = self.headers.copy()
                    
                    # form-data 형식으로 전송
                    form_data = {}
                    for key, value in data.items():
                        # Laravel 필드명에 맞게 수정
                        form_data[key] = (None, str(value))
                    
                    # 이미지 파일 추가
                    form_data.update(files)
                    
                    # 디버그 로그 추가
                    self.logger.debug(f"최종 전송 데이터: {[(k, v[0] if isinstance(v, tuple) else v) for k, v in form_data.items()]}")
                    
                    response = requests.post(
                        url, 
                        headers=headers,
                        files=form_data,
                        timeout=30
                    )
                    
                    # 응답 상태 코드 로깅
                    self.logger.debug(f"응답 상태 코드: {response.status_code}")
                    self.logger.debug(f"응답 헤더: {dict(response.headers)}")
                finally:
                    files.clear()
            else:
                self.logger.info(f"게시글 생성 시작 (이미지 없음) - 제목: {title}")
                payload = {
                    "title": title,
                    "content": content,
                    "category": category,
                    "writer": writer
                }
                response = requests.post(url, headers=self.headers, json=payload)

            # 응답 확인 및 한글 디코딩
            try:
                response.encoding = 'utf-8'  # 응답 인코딩을 UTF-8로 설정
                response_data = response.json()
                
                # 응답 로깅 (디버깅용)
                self.logger.debug(f"API 응답: {response_data}")
                
                if not response_data.get('success', False):
                    error_msg = f"게시글 생성 실패\n제목: {title}\n카테고리: {category}\n응답: {response.text}"
                    self.logger.error(error_msg)
                    raise ApiError(response.status_code, error_msg)

                self.logger.info(f"게시글 생성 성공 - 제목: {title}")
                
                # 이미지 URL 확인 로직 수정
                if image_paths and not response_data.get('data', {}).get('image_urls'):
                    self.logger.warning(f"이미지가 포함된 게시글이지만 image_urls가 비어있습니다. - 제목: {title}")
                elif image_paths:
                    self.logger.info(f"이미지 URL: {response_data.get('data', {}).get('image_urls')}")
                
                return response_data
                
            except ValueError as e:
                error_msg = f"JSON 응답 파싱 실패\n제목: {title}\n카테고리: {category}\n응답: {response.text}"
                self.logger.error(error_msg)
                raise ApiError(response.status_code, error_msg)

        except requests.RequestException as e:
            error_msg = f"API 요청 중 오류 발생\n제목: {title}\n카테고리: {category}\n오류: {str(e)}"
            self.logger.error(error_msg)
            raise ApiError(500, error_msg)

if __name__ == "__main__":
    # API 테스트
    api = ApiUtil()
    
    # 테스트할 이미지 경로
    image_paths = [
        "img/opm_kospi_20241209.jpg",
        "img/opm_kosdaq_20241209.jpg"
    ]
    
    # 테스트 데이터
    test_data = {
        "title": "API 이미지 전송 테스트",
        "content": "이미지 전송 테스트를 위한 게시글입니다.",
        "category": "거래량",
        "writer": "테스터"
    }
    
    try:
        # API 호출 테스트
        result = api.create_post(
            title=test_data["title"],
            content=test_data["content"],
            category=test_data["category"],
            writer=test_data["writer"],
            image_paths=image_paths
        )
        print("API 호출 결과:", result)
        
    except ApiError as e:
        print(f"API 에러 발생: {e}")
    except Exception as e:
        print(f"예상치 못한 에러 발생: {e}") 
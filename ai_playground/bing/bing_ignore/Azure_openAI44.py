import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, TypeVar, Generic, Tuple
from enum import Enum
from dataclasses import dataclass
import json
import os
import time

import aiohttp
from openai import AzureOpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 환경 변수에서 API 키 가져오기
AzureAIAssistant_api_key = os.getenv("AzureAIAssistant_api_key")
MET_data_source = os.getenv('MET_data_source')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('curation_log.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

T = TypeVar('T')

# 향상된 Result 타입
@dataclass
class Result(Generic[T]):
    """API 결과를 담는 일반화된 컨테이너"""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None

# 큐레이션 스타일 정의
class CurationStyle(Enum):
    EMOTIONAL = "Emotional"
    INTERPRETIVE = "Interpretive"
    HISTORICAL = "Historical"
    CRITICAL = "Critical"
    NARRATIVE = "Narrative"
    CONTEMPORARY_ART_CRITIC = "Contemporary_Art_Critic"
    ART_APPRAISER = "Art_Appraiser"
    AESTHETIC_EVALUATION = "Aesthetic_Evaluation"
    IMAGE_INTERPRETER = "Image_Interpreter"
    EDUCATIONAL = "Educational"

# AzureAIAssistant 클래스 (통합 버전)
class AzureAIAssistant:
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AzureAIAssistant_endpoint"),
            api_key=os.getenv("AzureAIAssistant_api_key"),
            api_version=os.getenv("AzureAIAssistant_api_version")
        )
        # 기존 어시스턴트 목록에서 첫 번째 어시스턴트 사용
        assistants = self.client.beta.assistants.list()
        self.assistant_id = assistants.data[0].id if assistants.data else None
        logger.info(f"사용할 어시스턴트 ID: {self.assistant_id}")
        
        # API 클라이언트 초기화
        self.api_client = ChatCompletionsClient(
            endpoint=os.getenv('OPENAI_ENDPOINT'),
            credential=AzureKeyCredential(AzureAIAssistant_api_key)
        )

    def wait_for_run_completion(self, thread_id, run_id, max_wait_time=60):
        """
        런 완료를 기다리는 메서드
        """
        start_time = time.time()
        print("분석 중", end="", flush=True)
        while True:
            # 런 상태 확인
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )
            # 런 상태에 따른 처리
            if run.status == "completed":
                print("\n분석 완료!")
                return True
            elif run.status in ["failed", "cancelled"]:
                print(f"\n분석 중 오류 발생: {run.status}")
                return False
            # 최대 대기 시간 초과 확인
            if time.time() - start_time > max_wait_time:
                print("\n분석 시간 초과")
                return False
            # 로딩 애니메이션
            print(".", end="", flush=True)
            time.sleep(1)

    def get_last_assistant_message(self, thread_id):
        """
        스레드의 마지막 어시스턴트 메시지 가져오기
        """
        messages = self.client.beta.threads.messages.list(thread_id=thread_id)
        for msg in reversed(messages.data):
            if msg.role == "assistant":
                return msg.content[0].text.value
        return None

    async def call_action(self, task, limit=5):
        """
        작업 처리를 위한 통합 메서드 (Assistant와 AzureAIAssistant 통합)
        """
        try:
            logger.info(f"실행 작업: {task}")
            
            # 작업이 "get_artwork_details_"로 시작하는 경우 (object_id 처리)
            if task.startswith("get_artwork_details_"):
                object_id = task.split("get_artwork_details_")[1]
                return await self.get_artwork_details(object_id)
            
            # 일반 검색 작업인 경우
            return await self.search_artworks(task, limit)
        except Exception as e:
            logger.error(f"작업 실행 오류: {e}")
            return Result(success=False, error=str(e))

    async def search_artworks(self, query, limit=5):
        """
        작품 검색 메서드 - MET API를 사용하여 작품 검색
        """
        try:
            # 여기서는 시뮬레이션된 응답을 반환합니다
            # 실제 구현에서는 MET API에 비동기 요청을 보냅니다
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{MET_data_source}/search?q={query}&hasImages=true") as response:
                    if response.status == 200:
                        data = await response.json()
                        # objectIDs가 없는 경우 빈 리스트 반환
                        object_ids = data.get("objectIDs", [])[:limit]
                        return Result(success=True, data={"query": query, "objectIDs": object_ids})
                    else:
                        return Result(success=False, error=f"API 오류: {response.status}")
        except Exception as e:
            logger.error(f"작품 검색 오류: {e}")
            return Result(success=False, error=str(e))

    async def get_artwork_details(self, object_id):
        """
        작품 상세 정보 가져오기 메서드
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{MET_data_source}/objects/{object_id}") as response:
                    if response.status == 200:
                        data = await response.json()
                        return Result(success=True, data=data)
                    else:
                        return Result(success=False, error=f"API 오류: {response.status}")
        except Exception as e:
            logger.error(f"작품 상세 정보 가져오기 오류: {e}")
            return Result(success=False, error=str(e))

    def test_agent(self, artwork_name, debug=False):
        """
        에이전트 테스트 함수
        Args:
        artwork_name (str): 분석할 예술 작품명
        debug (bool): 디버그 모드 활성화 여부
        """
        try:
            # 디버그 모드일 경우에만 상세 로깅
            if debug:
                logging.getLogger().setLevel(logging.INFO)
            # 스레드 생성
            thread = self.client.beta.threads.create()
            # 메시지 생성
            message = self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=f"Provide a comprehensive analysis of the artwork '{artwork_name}'"
            )
            # 스레드 런 생성
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant_id
            )
            # 런 완료 대기
            if self.wait_for_run_completion(thread.id, run.id):
                # 마지막 어시스턴트 메시지 가져오기
                analysis = self.get_last_assistant_message(thread.id)
                return {
                    "artwork": artwork_name,
                    "analysis": analysis
                }
            return None
        except Exception as e:
            # 디버그 모드일 경우에만 전체 오류 출력
            if debug:
                print(f"에이전트 테스트 중 오류 발생: {e}")
            return None

# 기본 API 클라이언트
class BaseAPIClient:
    def __init__(self, session: aiohttp.ClientSession, base_url: str):
        self.session = session
        self.base_url = base_url
        self.timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout

    async def make_request(self, method: str, endpoint: str, **kwargs) -> Result:
        """
        Make an HTTP request to the API
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            **kwargs: Additional arguments for the request
            
        Returns:
            Result object with response data or error
        """
        url = f"{self.base_url}/{endpoint}"
        logger.info(f"Making {method} request to {url}")
        
        try:
            async with self.session.request(
                method, 
                url, 
                timeout=self.timeout,
                **kwargs
            ) as response:
                return await self.handle_response(response)
        except asyncio.TimeoutError:
            logger.error(f"Request timeout: {url}")
            return Result(success=False, error="Request timeout")
        except aiohttp.ClientError as e:
            logger.error(f"ClientError in request to {url}: {str(e)}")
            return Result(success=False, error=f"Connection error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in request to {url}: {str(e)}")
            return Result(success=False, error=str(e))

    async def handle_response(self, response: aiohttp.ClientResponse) -> Result:
        """
        Handle the API response
        
        Args:
            response: aiohttp ClientResponse object
            
        Returns:
            Result object with parsed data or error
        """
        try:
            if response.status in (200, 201, 202):  # Success status codes
                # Try to parse as JSON
                try:
                    data = await response.json()
                    return Result(success=True, data=data)
                except json.JSONDecodeError:
                    # If not JSON, get text
                    text = await response.text()
                    return Result(success=True, data=text)
            elif response.status == 404:
                return Result(success=False, error="Resource not found")
            elif response.status == 429:
                return Result(success=False, error="Rate limit exceeded")
            elif 400 <= response.status < 500:
                error_text = await response.text()
                return Result(success=False, error=f"Client error: {response.status} - {error_text}")
            elif 500 <= response.status < 600:
                return Result(success=False, error=f"Server error: {response.status}")
            else:
                return Result(success=False, error=f"Unexpected status code: {response.status}")
        except Exception as e:
            logger.error(f"Error handling response: {str(e)}")
            return Result(success=False, error=f"Error processing response: {str(e)}")

# 향상된 데이터 모델
@dataclass
class ImageAnalysis:
    dense_caption: str
    tags: List[str]
    confidence_score: float

    def to_dict(self) -> Dict:
        return {
            "dense_caption": self.dense_caption,
            "tags": self.tags,
            "confidence_score": self.confidence_score
        }

@dataclass
class ArtReference:
    title: str
    artist: str
    period: str
    medium: str
    description: str
    url: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "artist": self.artist,
            "period": self.period,
            "medium": self.medium,
            "description": self.description,
            "url": self.url
        }

@dataclass
class CurationRequest:
    user_prompt: str
    selected_style: CurationStyle
    image_analysis: ImageAnalysis
    reference_search: Optional[str] = None
    language: str = 'ko'  # 언어 설정

    def to_dict(self) -> Dict:
        return {
            "user_prompt": self.user_prompt,
            "selected_style": self.selected_style.value,
            "image_analysis": self.image_analysis.to_dict(),
            "reference_search": self.reference_search,
            "language": self.language
        }

@dataclass
class CurationResult:
    style: CurationStyle
    content: str
    references: List[ArtReference]
    metadata: Dict
    language: str = 'ko'  # 언어 설정

    def to_dict(self) -> Dict:
        return {
            "style": self.style.value,
            "content": self.content,
            "references": [ref.to_dict() for ref in self.references],
            "metadata": self.metadata,
            "language": self.language
        }

class EnhancedCurationService:
    def __init__(self, met_client, gpt_client, session=None):
        """
        Initialize the EnhancedCurationService with required clients
        
        Args:
            met_client: Client for Metropolitan Museum API
            gpt_client: Client for GPT/Azure AI API
            session: Optional aiohttp ClientSession (can be passed from parent)
        """
        self.met_client = met_client
        self.gpt_client = gpt_client
        self.session = session
        self._init_style_prompts()

    def _init_style_prompts(self):
        """스타일별 프롬프트 템플릿 로드와 참조 데이터 요구사항 정의"""
        self.style_prompts = {
            CurationStyle.EMOTIONAL: {
                "prompt": """감성적이고 서정적인 관점에서 작품을 분석하여 다음 요소들을 포함해 서술해주세요:
                    - 작품이 전달하는 주요 감정과 분위기
                    - 시각적 요소가 불러일으키는 감정적 반응
                    - 작품 속 순간이 주는 특별한 정서
                    - 관객이 느낄 수 있는 공감과 울림
                    - 작품의 서정적 특징과 시적 표현""",
                "required_references": ["similar_emotional_works", "artist_background"]
            },
            CurationStyle.INTERPRETIVE: {
                "prompt": """작품의 의미와 예술적 기법을 심층적으로 분석하여 다음 요소들을 포함해 서술해주세요:
                    - 작품의 주요 시각적 요소와 상징성
                    - 구도와 색감의 효과
                    - 작가의 의도와 메시지
                    - 사용된 예술적 기법과 그 효과
                    - 작품이 전달하는 철학적/개념적 의미""",
                "required_references": ["artistic_techniques", "symbolism_history"]
            },
            CurationStyle.HISTORICAL: {
                "prompt": """작품을 역사적, 미술사적 맥락에서 심도 있게 분석하여 다음 요소들을 포함해 서술해주세요:
                    - 작품이 제작된 시대적 배경과 특징
                    - 유사한 예술 경향이나 작품들과의 관계
                    - 현대 미술사에서의 위치와 의의
                    - 작품의 예술적/사회적 영향력
                    - 역사적 맥락에서의 작품 해석""",
                "required_references": ["historical_context", "art_movement_history"]
            },
            CurationStyle.CRITICAL: {
                "prompt": """전문적이고 균형 잡힌 시각으로 작품을 비평하여 다음 요소들을 포함해 서술해주세요:
                    - 작품의 기술적 완성도와 예술성
                    - 창의성과 혁신성 분석
                    - 강점과 개선 가능성
                    - 예술적 성취와 한계점
                    - 작품의 독창성과 차별성""",
                "required_references": ["contemporary_critiques", "technical_analysis"]
            },
            CurationStyle.NARRATIVE: {
                "prompt": """작품을 매력적인 이야기로 풀어내어 다음 요소들을 포함해 서술해주세요:
                    - 작품 속 장면의 생생한 묘사
                    - 등장 요소들 간의 관계와 이야기
                    - 작품 속 시간의 흐름과 변화
                    - 장면 속에 숨겨진 드라마와 서사
                    - 관객이 상상할 수 있는 전후 맥락""",
                "required_references": ["narrative_context", "literary_connections"]
            },
            CurationStyle.CONTEMPORARY_ART_CRITIC: {
                "prompt": """현대 예술 트렌드의 관점에서 작품을 분석하여 다음 요소들을 포함해 서술해주세요:
                    - 현대 예술 동향과의 연관성
                    - 디지털/기술적 혁신 요소
                    - 현대 사회/문화적 맥락에서의 의미
                    - 최신 예술 트렌드와의 접점
                    - 미래 예술 발전에 대한 시사점""",
                "required_references": ["contemporary_trends", "digital_art_context"]
            },
            CurationStyle.ART_APPRAISER: {
                "prompt": """현대 예술 트렌드의 관점에서 작품을 분석하여 다음 요소들을 포함해 서술해주세요:
                    - 현대 예술 동향과의 연관성
                    - 디지털/기술적 혁신 요소
                    - 현대 사회/문화적 맥락에서의 의미
                    - 최신 예술 트렌드와의 접점
                    - 미래 예술 발전에 대한 시사점""",
                "required_references": ["contemporary_trends", "digital_art_context"]
            },
            CurationStyle.AESTHETIC_EVALUATION: {
                "prompt": """현대 미술에 대한 깊은 애정과 이해를 가진 열정적인 미술 옹호자로서, 다음 요소들을 고려하여 작품을 긍정적이고 영감을 주는 방식으로 분석해주세요:
                    - 작품의 혁신적 측면과 독창성
                    - 뛰어난 색채와 구도의 활용
                    - 작가의 비전과 그 탁월한 표현
                    - 관객에게 미치는 감정적, 지적 영향
                    - 현대 미술사적 맥락에서의 중요성
                    작품의 장점을 강조하고 예술적 가치를 생생하게 설명해주세요.""",
                "required_references": ["contemporary_trends", "digital_art_context"]
            },
            CurationStyle.IMAGE_INTERPRETER: {
                "prompt": """시각장애인을 위한 이미지 설명 전문가로서, 다음 요소들을 포함하여 명확하고 상세한 설명을 제공해주세요:
                    - 이미지의 전체적인 구성과 주요 요소
                    - 색상, 형태, 질감의 상세한 묘사
                    - 요소들 간의 공간적 관계와 배치
                    - 이미지가 전달하는 분위기나 감정
                    - 중요한 세부사항이나 특징적인 요소
                    촉각적 또는 청각적 경험과 연관지어 설명해주세요.""",
                "required_references": ["contemporary_trends", "digital_art_context"]
            },
            CurationStyle.EDUCATIONAL: {
                "prompt": """교육적 관점에서 작품을 분석하여 다음 요소들을 포함해 서술해주세요:
                    - 작품의 긍정적 측면과 성취된 학습 목표
                        * 기술적 완성도와 성공적인 표현 요소
                        * 효과적으로 전달된 메시지나 의도
                        * 창의적 시도와 혁신적 접근

                    - 발전 가능한 영역과 학습 제안
                        * 더 발전시킬 수 있는 기술적 요소
                        * 보완하면 좋을 표현적 측면
                        * 시도해볼 수 있는 새로운 접근방식

                    - 구체적인 학습 목표와 실천 방안
                        * 단기적으로 향상시킬 수 있는 부분
                        * 장기적인 발전을 위한 학습 방향
                        * 참고할 만한 작품이나 기법 추천

                    긍정적인 피드백을 중심으로, 발전 가능성을 구체적으로 제시해주세요.""",
                "required_references": ["contemporary_trends", "digital_art_context"]
            }
        }

    async def generate_curation(self, request: CurationRequest) -> Result[CurationResult]:
        try:
            # 병렬로 참조 작품과 컨텍스트 데이터 수집
            ref_artworks_task = self._search_reference_artworks(request)
            context_data_task = self._search_additional_context(request)

            ref_artworks, context_data = await asyncio.gather(
                ref_artworks_task,
                context_data_task
            )

            if not ref_artworks.success:
                return Result(success=False, error=ref_artworks.error)

            # 데이터 통합 및 큐레이션 생성
            integrated_data = self._integrate_data(request, ref_artworks.data, context_data.data)
            curation_text = await self._generate_gpt_curation(integrated_data)

            if not curation_text.success:
                return Result(success=False, error=curation_text.error)

            # 결과 포맷팅
            result = self._format_results(
                curation_text.data,
                ref_artworks.data,
                request
            )
            return Result(success=True, data=result)
        except Exception as e:
            logger.error(f"큐레이션 생성 중 오류 발생: {e}")
            return Result(success=False, error=str(e))

    async def _search_reference_artworks(self, request: CurationRequest) -> Result[List[Dict]]:
        """참조 작품 검색 - 비동기 처리"""
        try:
            # Get required reference types for the selected style
            style_refs = self.style_prompts[request.selected_style]["required_references"]
            search_tasks = []
            
            # Create search tasks for each reference type
            for ref_type in style_refs:
                # Use reference_search if available, otherwise use general terms
                search_term = request.reference_search if request.reference_search else "art"
                search_query = f"{search_term} {ref_type.replace('_', ' ')}"
                search_tasks.append(self.met_client.search_artworks(search_query, True))

            # Execute all search tasks in parallel
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            processed_results = []
            for result in search_results:
                # Skip exceptions
                if isinstance(result, Exception):
                    logger.warning(f"Error in search: {str(result)}")
                    continue
                    
                # Process successful Result objects
                if isinstance(result, Result) and result.success:
                    if isinstance(result.data, dict) and result.data.get("objectIDs"):
                        # Limit to top 3 objects for each search
                        object_ids = result.data.get("objectIDs", [])[:3]
                        
                        # Get details for each artwork in parallel
                        artwork_tasks = [
                            self.met_client.get_artwork_details(UnifiedInput(object_id=object_id))
                            for object_id in object_ids
                        ]
                        artwork_details = await asyncio.gather(*artwork_tasks, return_exceptions=True)
                        
                        # Add successful results
                        for detail in artwork_details:
                            if isinstance(detail, Result) and detail.success:
                                processed_results.append(detail.data)
                    # Handle string results (convert to dict)
                    elif isinstance(result.data, str):
                        try:
                            # Try to parse as JSON first
                            json_data = json.loads(result.data)
                            processed_results.append(json_data)
                        except json.JSONDecodeError:
                            # If not JSON, create a simple description dict
                            processed_results.append({
                                "title": "Reference",
                                "artist": "Unknown",
                                "period": "Unknown",
                                "medium": "Unknown",
                                "description": result.data
                            })
            
            # If we couldn't find any reference artworks, create a placeholder
            if not processed_results:
                processed_results.append({
                    "title": "No specific reference found",
                    "artist": "Various artists",
                    "period": "Various periods",
                    "medium": "Various media",
                    "description": f"General reference for {request.selected_style.value} style"
                })
                
            return Result(success=True, data=processed_results)
        except Exception as e:
            logger.error(f"Error in _search_reference_artworks: {str(e)}")
            return Result(success=False, error=str(e))

    async def _search_additional_context(self, request: CurationRequest) -> Result[Dict]:
        """추가 컨텍스트 정보 검색 - 비동기 처리"""
        try:
            context_data = {}
            style_context = self.style_prompts[request.selected_style]["required_references"]

            async def fetch_context(context_type: str) -> Tuple[str, Any]:
                # Use reference_search if available, otherwise use the context type
                search_term = request.reference_search if request.reference_search else "art history"
                search_query = f"{search_term} {context_type.replace('_', ' ')}"
                
                # Call the MetMuseum client for search
                result = await self.met_client.search_artworks(search_query)
                
                # Process the result
                if isinstance(result, Result) and result.success:
                    if isinstance(result.data, dict):
                        return context_type, result.data
                    elif isinstance(result.data, str):
                        try:
                            return context_type, json.loads(result.data)
                        except json.JSONDecodeError:
                            return context_type, {"description": result.data}
                
                # Default empty result if something went wrong
                return context_type, {"description": f"No data found for {context_type}"}

            # 병렬로 컨텍스트 데이터 수집
            context_tasks = [fetch_context(context_type) for context_type in style_context]
            results = await asyncio.gather(*context_tasks, return_exceptions=True)
            
            # Process results, handling exceptions
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Error fetching context: {str(result)}")
                    continue
                    
                if isinstance(result, tuple) and len(result) == 2:
                    context_type, data = result
                    context_data[context_type] = data

            return Result(success=True, data=context_data)
        except Exception as e:
            logger.error(f"Error in _search_additional_context: {str(e)}")
            return Result(success=False, error=str(e))

    async def _generate_assistant_response(self, prompt: str) -> str:
        """어시스턴트 응답 생성"""
        try:
            # 스레드 생성
            thread = self.gpt_client.client.beta.threads.create()
            logger.info(f"새 스레드 생성: {thread.id}")
            
            # 메시지 생성
            message = self.gpt_client.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=prompt
            )
            logger.info(f"스레드 메시지 생성: {message.id}")
            
            # 스레드 런 생성
            run = self.gpt_client.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.gpt_client.assistant_id
            )
            logger.info(f"스레드 런 생성: {run.id}")
            
            # 런 완료 대기
            if self.gpt_client.wait_for_run_completion(thread.id, run.id):
                # 마지막 어시스턴트 메시지 가져오기
                response = self.gpt_client.get_last_assistant_message(thread.id)
                return response if response else "No response generated"
            return "Response generation timed out"
        except Exception as e:
            logger.error(f"어시스턴트 응답 생성 중 오류 발생: {e}")
            return f"Error generating response: {str(e)}"


    async def _search_additional_context(self, request: CurationRequest) -> Result[Dict]:
        """추가 컨텍스트 정보 검색 - 비동기 처리"""
        try:
            context_data = {}
            style_context = self.style_prompts[request.selected_style]["required_references"]
            
            # 검색 쿼리 구성
            search_base = request.reference_search if request.reference_search else request.user_prompt
            
            # 컨텍스트 유형별로 쿼리 생성 및 처리
            context_tasks = {}
            for context_type in style_context:
                search_query = f"{search_base} {context_type.replace('_', ' ')} art"
                context_tasks[context_type] = self.assistant.call_action(search_query, limit=5)
            
            # 비동기 실행 및 결과 처리
            results = {}
            for context_type, task in context_tasks.items():
                result = await task
                
                if result.success:
                    if isinstance(result.data, dict):
                        # API 응답에서 필요한 데이터 추출
                        results[context_type] = self._extract_context_info(result.data, context_type)
                    elif isinstance(result.data, str):
                        # 문자열 결과를 그대로 사용
                        results[context_type] = result.data
                    else:
                        # 기본값 설정
                        results[context_type] = f"No specific {context_type} information available"
                else:
                    # 실패한 경우 기본값 설정
                    results[context_type] = f"Failed to retrieve {context_type} information"
            
            return Result(success=True, data=results)
        except Exception as e:
            logger.error(f"추가 컨텍스트 검색 중 오류 발생: {e}")
            return Result(success=False, error=str(e))

    def _extract_context_info(self, data, context_type):
        """컨텍스트 데이터에서 필요한 정보 추출"""
        try:
            # 여기서는 샘플로 단순화된 추출 로직을 제공합니다
            # 실제 데이터 형식에 맞게 수정이 필요할 수 있습니다
            if "objectIDs" in data and data["objectIDs"]:
                return f"Found {len(data['objectIDs'])} items related to"
            
            
            
###########################################################
# Fixed AsyncCurationClient for proper context management
class AsyncCurationClient:
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        self.met_client = MetMuseumClient(self.session)
        self.gpt_client = AzureAIAssistant()
        self.curation_service = EnhancedCurationService(
            met_client=self.met_client,
            gpt_client=self.gpt_client,
            session=self.session
        )
        return self.curation_service

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()

# Improved main function
async def main():
    # Configure logging with more detailed levels for debugging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('curation_log.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Check for required environment variables
    required_env_vars = ["AzureAIAssistant_api_key", "AzureAIAssistant_endpoint", 
                        "AzureAIAssistant_api_version", "MET_data_source"]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        print(f"Error: Missing environment variables: {', '.join(missing_vars)}")
        return
    
    try:
        # Create the curation request
        request = CurationRequest(
            user_prompt="in the style of Vincent van Gogh's Starry Night, oil painting",
            selected_style=CurationStyle.CRITICAL,
            image_analysis=ImageAnalysis(
                dense_caption="oil painting of a night street with a cafe in the style of John Singer Sargent, with lights, people walking around, bright moonlight, low angle, expressive brushstrokes, detailed background to add depth --ar 63:128 --stylize 750 --v 6.1",
                tags=["art", "painting"],
                confidence_score=0.95
            ),
            reference_search="van Gogh Starry Night",  # More specific reference search
            language='ko'  # 언어 설정
        )
        
        logger.info(f"Starting curation process for: {request.user_prompt}")
        print(f"Beginning art curation for: {request.user_prompt}")
        print(f"Selected style: {request.selected_style.value}")
        print(f"Reference search: {request.reference_search}")
        
        # Use the AsyncCurationClient context manager
        async with AsyncCurationClient() as curation_service:
            # Generate the curation
            print("Generating curation...")
            result = await curation_service.generate_curation(request)
            
            # Handle the result
            if result.success and result.data:
                print("\n===== Curation Result =====")
                result_dict = result.data.to_dict()
                
                # Print style and language
                print(f"Style: {result_dict['style']}")
                print(f"Language: {result_dict['language']}")
                
                # Print content with line breaks for readability
                print("\n----- Content -----")
                content_lines = result_dict['content'].split(". ")
                for line in content_lines:
                    if line.strip():
                        print(f"{line.strip()}.")
                
                # Print references
                print("\n----- References -----")
                for i, ref in enumerate(result_dict['references'], 1):
                    print(f"{i}. {ref['title']} by {ref['artist']} ({ref['period']})")
                
                # Print metadata
                print("\n----- Metadata -----")
                print(f"Timestamp: {result_dict['metadata']['timestamp']}")
                print(f"Reference count: {result_dict['metadata']['reference_count']}")
                
            else:
                error_message = result.error if hasattr(result, 'error') else "Unknown error"
                logger.error(f"Curation generation failed: {error_message}")
                print(f"Error generating curation: {error_message}")
    
    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}")
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
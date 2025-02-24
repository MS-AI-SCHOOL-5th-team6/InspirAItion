# bing / Azure_openAI.py

##########################################################
# Azure AI Projects 프레임워크를 사용한 기본 도구 구조
# Azure OpenAI에 대한 인증 설정
# Metropolitan Museum API에 대한 기본 API 연결
# Pydantic을 사용한 입력 검증 스키마
# 기본 로깅 구성
# 두 가지 주요 기능을 갖춘 도구 정의 인터페이스
#  ├ search_artworks: 컬렉션에서 아트워크 검색
#  └ get_artwork_details: 특정 아트워크에 대한 자세한 정보
##########################################################

"""
DESCRIPTION:
    MET Museum API integration for Azure AI Studio
    Allows searching artworks and retrieving artwork details from the Metropolitan Museum of Art

USAGE:
    Requires .env file with:
    - PROJECT_CONNECTION_STRING: Azure AI Project connection string
    - PROJECT_API_KEY: Project API key
    - MODEL_DEPLOYMENT_NAME: AI model deployment name
"""

# 표준 라이브러리
import logging
import os
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from dataclasses import dataclass
import json

# 서드파티 라이브러리
from dotenv import load_dotenv
import requests
from pydantic import BaseModel, Field
from azure.ai.projects import AIProjectClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.projects.models import Tool
from openai import AzureOpenAI
from azure.ai.inference import ChatCompletionsClient
from requests.exceptions import RequestException

# 프로젝트 관련 라이브러리



# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('art_curation_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Azure OpenAI client
# 하드코딩된 API 키
api_key = '883MBnccg0TLFV7MEZNVqmFRwmEiEBx0SbBiivVnEIefJgVkW4JTJQQJ99BBACHYHv6XJ3w3AAAAACOGmcwh'

# openAI client
client = ChatCompletionsClient(
    endpoint='https://openaio3team64150034964.services.ai.azure.com',
    credential=AzureKeyCredential(api_key)
)

# Azure openAI client
Azure_openAI_client = AzureOpenAI(azure_endpoint="https://openaio3team64150034964.openai.azure.com/",
                api_key="883MBnccg0TLFV7MEZNVqmFRwmEiEBx0SbBiivVnEIefJgVkW4JTJQQJ99BBACHYHv6XJ3w3AAAAACOGmcwh",
                api_version="2024-05-01-preview")






# AI 응답을 받아오는 함수 설정
def get_ai_response(prompt: str) -> str:
    """AI 응답을 받아오는 함수"""
    try:
        response = client.chat.completions.create(
            model=os.getenv("MODEL_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": "You are an art expert assistant that can analyze and explain artworks from the Metropolitan Museum of Art."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"AI 응답 생성 중 오류 발생: {e}")
        return f"AI 응답 생성 실패: {str(e)}"

# 통합된 입력 구조
class UnifiedInput(BaseModel):
    query: Optional[str] = Field(default=None, description="Search query for artworks")
    has_images: Optional[bool] = Field(default=True, description="Filter for artworks with images")
    object_id: Optional[int] = Field(default=None, description="The ID of the artwork to get details for")
    artwork_title: Optional[str] = Field(default=None, description="Title of the artwork to search for")
    artist_name: Optional[str] = Field(default=None, description="Artist name to include in search")
    search_type: Optional[str] = Field(default="artwork", description="Type of search (artwork, artist, historical_context, artwork_analysis)")

# MetMuseumTool 클래스
# MET Museum의 API를 사용해 작품을 검색하고 상세 정보를 가져오는데 사용
class MetMuseumTool(Tool):
    def __init__(self):
        super().__init__()
        self.base_url = "https://collectionapi.metmuseum.org/public/collection/v1"
        self.bing_search_endpoint = os.getenv("AZURE_BING_ENDPOINT","").strip()
        self.bing_subscription_key = os.getenv("AZURE_BING_SUBSCRIPTION_KEY", "").strip()

        if not self.bing_search_endpoint or not self.bing_subscription_key:
            logger.warning("Bing Search credentials not fully configured")

    def definitions(self) -> List[Dict[str, Any]]:
        """Tool의 기능 정의"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_artworks",
                    "description": "메트로폴리탄 박물관 컬렉션에서 작품 검색",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "검색어"
                            },
                            "has_images": {
                                "type": "boolean",
                                "description": "이미지 있는 작품만 필터링"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_artwork_details",
                    "description": "특정 작품의 상세 정보와 분석 조회",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "object_id": {
                                "type": "integer",
                                "description": "작품 ID"
                            }
                        },
                        "required": ["object_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_artwork_context",
                    "description": "Bing을 사용한 작품 관련 추가 정보 검색",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "artwork_title": {
                                "type": "string",
                                "description": "작품 제목"
                            },
                            "artist_name": {
                                "type": "string",
                                "description": "작가 이름"
                            },
                            "search_type": {
                                "type": "string",
                                "enum": ["artwork", "artist", "historical_context"],
                                "description": "검색 유형"
                            }
                        },
                        "required": ["artwork_title"]
                    }
                }
            }
        ]

    def resources(self) -> List[str]:
        """Define any additional resources or dependencies"""
        return [
            "https://collectionapi.metmuseum.org/public/collection/v1"
        ]

    def execute(self, function_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """함수 실행"""
        try:
            if function_name == "search_artworks":
                return self.search_artworks(UnifiedInput(**parameters))
            elif function_name == "get_artwork_details":
                return self.get_artwork_details(UnifiedInput(**parameters))
            elif function_name == "search_artwork_context":
                return self.search_artwork_context(UnifiedInput(**parameters))
            else:
                raise ValueError(f"Unknown function: {function_name}")
        except Exception as e:
            logger.error(f"Function execution error: {e}")

    # search_artworks 메서드 ( 입력 SearchArtworksInput = Pydantic 모델 )
    # 필드 : query (str), has_images (Optional[bool])
    def search_artworks(self, input: UnifiedInput) -> dict:
        """Search for artworks in the Metropolitan Museum collection"""
        try:
            url = f"{self.base_url}/search"
            params = {
                "q": input.query,
                "hasImages": input.has_images
            }
            logger.info(f"Searching artworks with query: {input.query}")

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()  # HTTP 에러 발생 시 예외 처리

            result = response.json()
            logger.info(f"Search results: {len(result.get('objectIDs', []))} artworks found")
            return result

        except RequestException as e:
            logger.error(f"API 요청 중 오류 발생: {e}")
            return {
                "error": True,
                "message": f"Network error: {str(e)}"
            }
        except ValueError as e:
            logger.error(f"JSON 파싱 중 오류 발생: {e}")
            return {
                "error": True,
                "message": f"JSON parsing error: {str(e)}"
            }
        except Exception as e:
            logger.error(f"예기치 않은 오류 발생: {e}")
            return {
                "error": True,
                "message": f"Unexpected error: {str(e)}"
            }

    # get_artwork_details 메서드 ( 입력 ArtworkDetailsInput = Pydantic 모델)
    # 필드: object_id (int)
    def get_artwork_details(self, input: UnifiedInput) -> dict:
        """Get detailed information about a specific artwork"""
        try:
            url = f"{self.base_url}/objects/{input.object_id}"
            logger.info(f"Fetching details for artwork ID: {input.object_id}")

            response = requests.get(url, timeout=10)
            response.raise_for_status()

            result = response.json()

            # 화풍과 그림 분석 추가
            '''
            artwork_analysis = self._analyze_artwork(result)
            result['analysis'] = artwork_analysis

            if "historical_context" in input.search_type:
                context_data = self._get_historical_context(input.object_id)
                result['historical_context'] = context_data
            '''
            
            logger.info(f"Artwork details: {result}")
            return result

        except RequestException as e:
            logger.error(f"API 요청 중 오류 발생: {e}")
            return {
                "error": True,
                "message": f"Network error: {str(e)}"
            }
        except ValueError as e:
            logger.error(f"JSON 파싱 중 오류 발생: {e}")
            return {
                "error": True,
                "message": f"JSON parsing error: {str(e)}"
            }
        except Exception as e:
            logger.error(f"예기치 않은 오류 발생: {e}")
            return {
                "error": True,
                "message": f"Unexpected error: {str(e)}"
            }
            

# class AzureAIAssistant Tool
class AzureAIAssistant:
    def __init__(self, 
                azure_endpoint="https://openaio3team64150034964.openai.azure.com/",
                api_key="883MBnccg0TLFV7MEZNVqmFRwmEiEBx0SbBiivVnEIefJgVkW4JTJQQJ99BBACHYHv6XJ3w3AAAAACOGmcwh",
                api_version="2024-05-01-preview"):
        """
        Azure OpenAI 어시스턴트 초기화
        
        Args:
            azure_endpoint (str): Azure OpenAI 엔드포인트 URL
            api_key (str): Azure OpenAI API 키
            api_version (str): API 버전
        """
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version
        )
        
        # 어시스턴트 목록에서 첫 번째 어시스턴트 ID 설정
        assistants = self.client.beta.assistants.list()
        self.assistant_id = assistants.data[0].id if assistants.data else None
        
        logger.info(f"사용할 어시스턴트 ID: {self.assistant_id}")

    def wait_for_run_completion(self, thread_id, run_id, max_wait_time=60):
        """
        주어진 런 ID에 대해 런 상태를 확인하며, 최대 대기 시간 내에 완료 여부 확인
        
        Args:
            thread_id (str): 스레드 ID
            run_id (str): 런 ID
            max_wait_time (int): 최대 대기 시간 (초)
        
        Returns:
            bool: 런 완료 여부
        """
        start_time = time.time()
        
        while True:
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )
            
            if run.status == "completed":
                return True
            elif run.status in ["failed", "cancelled"]:
                logger.error(f"런 상태 오류: {run.status}")
                return False
            
            if time.time() - start_time > max_wait_time:
                logger.error("런 완료 대기 시간 초과")
                return False
            
            time.sleep(1)

    def get_last_assistant_message(self, thread_id):
        """
        스레드에서 마지막 어시스턴트 메시지를 가져옵니다
        
        Args:
            thread_id (str): 스레드 ID
        
        Returns:
            str: 어시스턴트의 마지막 메시지 내용
        """
        messages = self.client.beta.threads.messages.list(thread_id=thread_id)
        
        for msg in reversed(messages.data):
            if msg.role == "assistant":
                return msg.content[0].text.value
        
        return None

    def create_thread_and_run(self, user_message):
        """
        새로운 스레드를 생성하고, 사용자 메시지를 보내고, 런을 생성합니다
        
        Args:
            user_message (str): 사용자로부터 받은 메시지
        
        Returns:
            dict: 분석 결과 또는 None
        """
        try:
            # 스레드 생성
            thread = self.client.beta.threads.create()
            logger.info(f"새 스레드 생성: {thread.id}")

            # 사용자 메시지 전송
            message = self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=user_message
            )
            logger.info(f"스레드 메시지 생성: {message.id}")

            # 런 생성
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant_id
            )
            logger.info(f"스레드 런 생성: {run.id}")

            # 런 완료 대기
            if self.wait_for_run_completion(thread.id, run.id):
                analysis = self.get_last_assistant_message(thread.id)
                
                return {"analysis": analysis}
            
            return None

        except Exception as e:
            logger.error(f"에이전트 테스트 중 오류 발생: {e}")
            return None

'''
# 사용 예시
def main():
    ai_assistant = AzureAIAssistant()
    result = ai_assistant.create_thread_and_run("Provide a comprehensive analysis of the artwork 'Starry Night'")
    
    if result:
        print("🎨 예술 작품 분석 결과:")
        print(result['analysis'])
    else:
        print("분석 실패")

if __name__ == "__main__":
    main()
    '''

# 핵심 클래스와 데이터 구조
# 1. 큐레이션 스타일을 정의하는 Enum
class CurationStyle(Enum):
    EMOTIONAL = "Emotional"
    INTERPRETIVE = "Interpretive"
    HISTORICAL = "Historical"
    CRITICAL = "Critical"
    NARRATIVE = "Narrative" 
    Contemporary_Art_Critic = "Trend" # 미술 트렌드 분석가 # 현대 미술 비평가
    '''
    [Trend - 큐레이터 정의 후보]
    - Contemporary Art Critic: Analyzes and critiques artworks based on current trends and societal context.
    - Art Trend Analyst: Focuses on evaluating artworks in relation to contemporary art movements and trends.
    - Digital Art Curator: Specializes in curating and evaluating digital and technologically innovative artworks.
    - Cultural Art Consultant: Advises on the cultural relevance and societal impact of artworks. Assesses artworks with a focus on their implications for future art developments.
    
    - 현대 미술 비평가: 현재의 트렌드와 사회적 맥락을 바탕으로 작품을 분석하고 비평합니다.
    - 미술 트렌드 분석가: 현대 미술 운동과 트렌드와 관련하여 작품을 평가합니다.
    - 디지털 아트 큐레이터: 디지털 및 기술 혁신 작품을 전문적으로 큐레이팅하고 평가합니다.
    - 문화 예술 컨설턴트: 작품의 문화적 관련성과 사회적 영향을 조언합니다. 미래 예술 발전에 대한 함의를 중심으로 작품을 평가합니다.

    [ style 참고 ]

    - Trend-Based Analysis: Evaluates the artwork's alignment with current art trends.
    - Innovation-Focused Appraisal: Highlights the digital and technological elements of the artwork.
    - Socio-Cultural Evaluation: Considers the artwork's meaning within modern society and culture.
    - Trend-Connected Assessment: Examines the artwork's connection to the latest art trends.
    - Future-Oriented Valuation: Looks at the artwork's potential impact on future art developments.
    - Contextual Critique: Analyzes the artwork within the context of contemporary issues and movements.
    - Digital Innovation Review: Focuses on the technological advancements present in the artwork.
    - Cultural Relevance Appraisal: Assesses the artwork's significance in today's cultural landscape.
    - Trend Integration Analysis: Evaluates how well the artwork integrates with current art trends.
    - Forward-Looking Evaluation: Considers the implications of the artwork for the future of art.

    - 트렌드 기반 분석 ┬ 작품이 현재의 예술 트렌드와 얼마나 일치하는지 평가합니다. 작품의 디지털 및 기술적 요소를 강조합니다. 현대 사회와 문화 내에서 작품의 의미를 고려합니다.
                    └ 최신 예술 트렌드와의 연결성을 검토합니다.
    - 미래 지향 평가 ┬ 작품이 미래 예술 발전에 미칠 잠재적 영향을 살펴봅니다. 작품이 미래 예술에 미칠 함의를 고려합니다.
                    └ 작품에 존재하는 기술적 진보를 중점적으로 평가합니다(디지털 혁신 리뷰).
    - 문화적 관련성 감정: 오늘날의 문화적 풍경에서 작품의 중요성을 평가합니다.
    - 트렌드 통합 분석: 작품이 현재 예술 트렌드와 얼마나 잘 통합되는지 평가합니다. 현대 이슈와 운동의 맥락에서 작품을 분석합니다.
    
    '''
    
    Art_Appraiser = "Money" # 감정사
    '''
    감정사 (Art Appraiser)
    - Value-Based Assessment: Focuses on the intrinsic and market value of the artwork.
    - Condition-Driven Evaluation: Considers the physical state and preservation of the piece.
    - Historical Analysis: Takes into account the artwork's provenance and historical significance.

    - 가치 기반 평가: 작품의 내재적 가치와 시장 가치를 중시합니다.
    - 상상태 중심 평가: 작품의 물리적 상태와 보존 상태를 고려합니다.
    - 역사적 분석: 작품의 출처와 역사적 중요성을 고려합니다.
    '''
    
    Aesthetic_Evaluation = "Praise" # 미학적 평가 # 예술적+찬미적 해석
    
    '''
    - "긍정적 미술 비평" (Positive Art Criticism): 작품의 우수성을 강조하고, 예술적 기여와 영향력을 다루는 방식.
    - "미술적 찬미적 해석" (Panegyric Interpretation in Art): 작품의 예술적 가치와 창작자의 독창성을 찬양하며, 작품이 미술사에서 차지하는 중요성을 탐구하는 접근.
    - "미학적 찬미" (Aesthetic Panegyric): 작품의 미적 요소를 중심으로 그 예술적 아름다움과 창의력을 찬미하는 해석.
    - "미학적 평가" (Aesthetic Evaluation): 작품의 미학적 특성을 중점적으로 분석하고, 그 예술적 가치를 찬미하는 접근.
    - "예술적 찬미적 해석" (Artistic Panegyric Interpretation): 예술 작품의 의미, 감동, 그리고 그 독창성을 강조하는 깊이 있는 해석.
    - "미술사적 찬미적 해석" (Art Historical Panegyric Interpretation): 작품이 미술사와 예술 발전에 끼친 영향력과 그 중요성을 강조하는 학문적 접근.
    '''
    
    Image_Interpreter = "Blind"  # 이미지 해석가 # 이미지 내레이터 # 접근성 있는 이미지 큐레이터
    '''
    [역할]
    - Image Interpreter: A person who translates visual information into descriptive language, making images accessible to visually impaired individuals.
    - Visual Description Specialist: A professional who provides clear and vivid descriptions of images, helping individuals imagine the visual content.
    - Tactile Experience Guide: A person who connects the visual world with tactile or auditory sensations, providing descriptions that relate to touch and sound.
    - Accessible Image Curator: A role focused on making visual art and imagery accessible to all by crafting detailed descriptions for the visually impaired.
    - Sensory Translator: A person who translates the visual experience into sensory details, focusing on textures, sounds, and shapes to offer a comprehensive understanding.
    - Image Narrator: A person who tells the story of an image, describing its composition, mood, and key details through a rich, narrative approach.
    - Contextual Visual Expert: A role that not only describes the image in detail but also provides context, such as emotional tone, cultural relevance, and purpose.
    - 이미지 해석가: 시각적 정보를 설명하여 시각장애인이 이미지를 상상할 수 있도록 돕는 역할.
    - 시각적 설명 전문가: 이미지의 색상, 형태, 질감 등을 명확하고 풍부하게 설명하여 시각장애인에게 이미지를 전달하는 전문가.
    - 촉각적 경험 안내자: 시각적 이미지를 촉각적 또는 청각적 경험과 연결하여 설명하는 역할.
    - 접근성 있는 이미지 큐레이터: 시각장애인을 위해 이미지나 미술 작품을 접근 가능하게 만드는 역할, 세밀한 설명을 제공.
    - 감각적 번역가: 시각적 요소를 감각적인 세부 사항으로 번역하여 이미지에 대한 종합적인 이해를 돕는 사람.
    - 이미지 내레이터: 이미지의 구성, 분위기, 주요 디테일을 이야기처럼 풀어 설명하는 역할.
    - 상황적 시각 전문가: 이미지를 세부적으로 설명하는 것뿐만 아니라, 감정적인 톤이나 문화적 맥락 등을 제공하여 이미지의 의미를 풀어내는 역할.
    
    [ 해석 스타일]
    - Detailed Sensory Description: Focuses on providing a comprehensive, sensory-based explanation, relating visual elements to tactile or auditory experiences.
    - Narrative Image Breakdown: Describes the image through a storytelling approach, emphasizing composition, mood, and key details that bring the image to life.
    - Contextual Explanation: A style that goes beyond visual details to include the broader meaning, cultural or emotional context, and purpose of the image.
    - Clear and Concise Visual Description: A straightforward style that ensures all important elements are explained simply but vividly, without overwhelming the listener.
    - Expressive Emotional Mapping: Focuses on conveying the mood or emotion of the image, helping the listener understand the emotional tone of the scene.
    - 세밀한 감각적 설명 (Detailed Sensory Description):시각적 요소를 촉각적이나 청각적 경험과 연결하여, 종합적이고 감각적인 설명을 제공하는 방식입니다.
    - 서사적 이미지 분석 (Narrative Image Breakdown):이미지를 이야기처럼 풀어 설명하는 방식으로, 구성, 분위기, 주요 디테일을 강조하여 이미지를 생동감 있게 전달합니다.
    - 상황적 설명 (Contextual Explanation):시각적 디테일을 넘어서서, 이미지의 더 넓은 의미나 문화적, 감정적 맥락을 포함한 설명을 제공하는 스타일입니다.
    - 명료하고 간결한 시각적 설명 (Clear and Concise Visual Description):모든 중요한 요소들을 간단하고 생동감 있게 설명하는 방식으로, 청중이 부담 없이 이해할 수 있도록 합니다.
    - 표현적 감정 지도 (Expressive Emotional Mapping):이미지의 분위기나 감정을 전달하는 데 집중하여, 장면의 감정적 톤을 청중이 이해할 수 있도록 돕는 스타일입니다.
    
    [ 예시 ]
    - English: "The image is of a calm sunset over a lake. The sky is painted with soft, warm colors—pinks and oranges blend together like a peaceful summer evening. At the center, there is a still body of water reflecting the sky’s hues, with the silhouette of a tree standing gracefully at the far right. The scene feels peaceful, serene, and quiet, like the last moments of daylight before nightfall."
    - Korean: "이 이미지는 호수 위로 펼쳐진 차분한 일몰을 담고 있습니다. 하늘은 부드럽고 따뜻한 색으로 물들어 있으며, 핑크와 오렌지 색이 여유로운 여름 저녁처럼 섞입니다. 가운데는 하늘의 색을 반영하는 고요한 물이 있으며, 오른쪽 끝에는 나무의 실루엣이 우아하게 서 있습니다. 이 장면은 평화롭고 조용한 느낌을 주며, 밤이 오기 전 마지막 빛을 경험하는 듯한 분위기를 전달합니다."
    '''
    
    EDUCATIONAL = "Educational"  # 새로운 교육적 큐레이션 스타일 추가
    
# 2. Azure Vision 분석 결과를 담는 데이터 클래스
@dataclass
class ImageAnalysis:
    dense_caption: str
    tags: List[str]
    confidence_score: float

# 3. 참조 데이터 담는 클래스스
@dataclass
class ArtReference:
    title: str
    artist: str
    period: str
    medium: str
    description: str
    url: Optional[str] = None

# 4. 큐레이션 생성 요청을 위한 데이터 클래스
@dataclass
class CurationRequest:
    user_prompt: str
    selected_style: CurationStyle
    image_analysis: ImageAnalysis
    reference_search: Optional[str] = None

# 5. 생성된 큐레이션 결과를 담는 데이터 클래스
@dataclass
class CurationResult:
    style: CurationStyle
    content: str
    references: List[ArtReference]
    metadata: Dict

# 참조 데이터 통합
class EnhancedCurationService:
    def __init__(self, gpt_client, vision_client, met_client, bing_client):
        self.timeout = 30                   # 30초 타임아웃 (외부 API 호출용, 기본 타임아웃)
        self.gpt_client = gpt_client
        self.vision_client = vision_client
        self.met_client = met_client         # MET Museum API 통합 관련 작품 검색
        self.bing_client = bing_client       # Bing Search 통한 예술사적 컨텍스트 수집
        self._load_style_prompts()           # 스타일별 맞춤형 참조 데이터 요구사항 정의

    # 스타일별 전문성 강화
        # 각 스타일에 필요한 특정 참조 데이터 정의
        # 스타일별 전문적 분석 프롬프트 구성
        # 맞춤형 컨텍스트 정보 수집

    def _load_style_prompts(self):
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
            
            CurationStyle.Contemporary_Art_Critic: {
                "prompt": """현대 예술 트렌드의 관점에서 작품을 분석하여 다음 요소들을 포함해 서술해주세요:
                    - 현대 예술 동향과의 연관성
                    - 디지털/기술적 혁신 요소
                    - 현대 사회/문화적 맥락에서의 의미
                    - 최신 예술 트렌드와의 접점
                    - 미래 예술 발전에 대한 시사점""",
                "required_references": ["contemporary_trends", "digital_art_context"]
            },
            CurationStyle.Art_Appraiser:{
                "prompt":"""현대 예술 트렌드의 관점에서 작품을 분석하여 다음 요소들을 포함해 서술해주세요:
                    - 현대 예술 동향과의 연관성
                    - 디지털/기술적 혁신 요소
                    - 현대 사회/문화적 맥락에서의 의미
                    - 최신 예술 트렌드와의 접점
                    - 미래 예술 발전에 대한 시사점""",
                "required_references": ["contemporary_trends", "digital_art_context"]
            },
            CurationStyle.Aesthetic_Evaluation:{
                "prompt":"""현대 미술에 대한 깊은 애정과 이해를 가진 열정적인 미술 옹호자로서, 다음 요소들을 고려하여 작품을 긍정적이고 영감을 주는 방식으로 분석해주세요:
                    - 작품의 혁신적 측면과 독창성
                    - 뛰어난 색채와 구도의 활용
                    - 작가의 비전과 그 탁월한 표현
                    - 관객에게 미치는 감정적, 지적 영향
                    - 현대 미술사적 맥락에서의 중요성
                    작품의 장점을 강조하고 예술적 가치를 생생하게 설명해주세요.""",
                "required_references": ["contemporary_trends", "digital_art_context"]
            },
            CurationStyle.Image_Interpreter:{
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
        
    async def generate_curation(self, request: CurationRequest) -> CurationResult:
        """큐레이션 생성의 전체 프로세스를 조율"""
        # 1. MET Museum API를 통한 참조 작품 검색
        ref_artworks = await self._search_reference_artworks(request)

        # 2. Bing Search를 통한 추가 컨텍스트 검색
        context_data = await self._search_additional_context(request, ref_artworks)

        # 3. 데이터 통합
        integrated_data = self.integrate_data(request, ref_artworks, context_data)

        # 4. GPT를 통한 큐레이션 텍스트 생성
        curation_text = await self._generate_gpt_curation(integrated_data)

        # 5. 참조 정보 구성
        references = self._compile_references(ref_artworks, context_data)

        # 6. 결과 포맷팅 및 반환
        return self.format_results(curation_text, references, request)

    async def _search_reference_artworks(self, request: CurationRequest) -> List[Dict]:
        """MET Museum API를 사용하여 참조할 만한 작품들을 검색"""
        # try-except 구문 = 개별 검색 실패에도 계속 진행
        try:
            style_refs = self.style_prompts[request.selected_style]["required_references"]
            search_results = []

            for ref_type in style_refs:
                # 개별 작품 정보 조회 실패 시에도 계속 진행
                try:
                    search_query = f"{request.reference_search} {ref_type.replace('_', ' ')}"
                    results = await self.met_client.search_artworks({"query": search_query})
                    
                    if results.get("objectIDs"):
                        for object_id in results["objectIDs"][:3]:  # 상위 3개만 가져오기
                            try:
                                artwork_details = await self.met_client.get_artwork_details({"object_id": object_id})
                                if artwork_details:
                                    search_results.append(artwork_details)
                            except Exception as e:
                                logging.error(f"Error getting artwork details for ID {object_id}: {e}")
                                continue
                # 각 에러 상황별 구체적인 로깅 메시지 추가
                except Exception as e:
                    logging.error(f"Error in searching artworks for ref_type {ref_type}: {e}")
                    continue

            return search_results
        # 프로세스가 완전히 중단되지 않고 가능한 만큼 결과 반환
        except Exception as e:
            logging.error(f"Critical error in reference artwork search: {e}")
            return []  # 빈 리스트 반환하여 전체 프로세스 중단 방지


    async def _search_additional_context(self, request: CurationRequest, ref_artworks: List[Dict]) -> Dict:
        """Bing Search를 사용하여 추가 컨텍스트 정보 검색"""
        context_data = {}

        # 스타일별 필요한 컨텍스트 정보 검색 및 맞춤 검색 쿼리
        style_context = self.style_prompts[request.selected_style]["required_references"]
        for context_type in style_context:
            search_query = f"{request.reference_search} {context_type.replace('_', ' ')} art history"
            results = await self.bing_client.search_artwork_context({
                "artwork_title": request.reference_search,
                "search_type": context_type
            })
            context_data[context_type] = results

        return context_data

    def integrate_data(self, request: CurationRequest, ref_artworks: List[Dict], context_data: Dict) -> Dict:
        """모든 수집된 데이터를 통합"""
        return {
            "user_prompt": request.user_prompt,
            "selected_style": request.selected_style,  # selected_style 추가
            "image_analysis": {
                "caption": request.image_analysis.dense_caption,
                "tags": request.image_analysis.tags,
                "confidence": request.image_analysis.confidence_score
            },
            "reference_search": request.reference_search,  # reference_search 추가
            "ref_artworks": ref_artworks,
            "context_data": context_data
        }

    def _compile_references(self, ref_artworks: List[Dict], context_data: Dict) -> List[ArtReference]:
        """참조 정보 컴파일"""
        references = []

        # 작품 참조 정보 추가
        for artwork in ref_artworks:
            references.append(ArtReference(
                title=artwork.get("title", ""),
                artist=artwork.get("artistDisplayName", ""),
                period=artwork.get("period", ""),
                medium=artwork.get("medium", ""),
                description=artwork.get("description", ""),
                url=artwork.get("primaryImage", "")
            ))

        # 컨텍스트 데이터에서 추가 참조 정보 추가
        for context_type, data in context_data.items():
            if "sources" in data:
                for source in data["sources"]:
                    references.append(ArtReference(
                        title=source.get("title", ""),
                        artist="",
                        period="",
                        medium="",
                        description=source.get("snippet", ""),
                        url=source.get("url", "")
                    ))

        return references

    def format_results(self, curation_text: str, references: List[ArtReference], request: CurationRequest) -> CurationResult:
        """최종 큐레이션 결과 포맷팅"""
        return CurationResult(
            style=request.selected_style,
            content=curation_text,
            references=references,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "prompt_length": len(request.user_prompt),
                "tag_count": len(request.image_analysis.tags),
                "reference_count": len(references),
                "style_specific_data": self.style_prompts[request.selected_style]["required_references"]
            }
        )


    async def _generate_gpt_curation(self, integrated_data: Dict) -> str:
        """GPT를 사용하여 큐레이션 텍스트를 생성합니다."""
        # 1. 통합된 데이터를 기반으로 GPT 프롬프트를 생성합니다.
        prompt = self._create_gpt_prompt(integrated_data)

        # 2. GPT 클라이언트를 사용하여 큐레이션 텍스트를 생성합니다.
        response = await self.gpt_client.generate_text(prompt)

        # 3. 생성된 텍스트를 반환합니다.
        return response['generated_text']


    def _create_gpt_prompt(self, integrated_data: Dict) -> str:
        """통합된 데이터를 기반으로 GPT 프롬프트를 생성합니다."""
        user_prompt = integrated_data['user_prompt']
        selected_style = integrated_data['selected_style']
        image_analysis = integrated_data['image_analysis']
        reference_search = integrated_data['reference_search']
        ref_artworks = integrated_data['ref_artworks']
        context_data = integrated_data['context_data']

        # 스타일별 프롬프트 템플릿을 가져옵니다.
        style_prompt = self.style_prompts[selected_style]["prompt"]

        # 프롬프트를 구성합니다.
        prompt = f"{style_prompt}\n\n"
        prompt += f"User Prompt: {user_prompt}\n"
        prompt += f"Image Analysis: {image_analysis}\n"
        prompt += f"Reference Search: {reference_search}\n"
        prompt += f"Reference Artworks: {ref_artworks}\n"
        prompt += f"Context Data: {context_data}\n"

        return prompt






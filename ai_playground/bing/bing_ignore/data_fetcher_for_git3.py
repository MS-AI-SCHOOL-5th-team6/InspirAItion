import os
import asyncio
import traceback
import logging
import requests
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv
from dataclasses import dataclass
from pydantic import BaseModel, Field
from openai import AzureOpenAI
from azure.ai.projects import AIProjectClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.projects.models import Tool
import time

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
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY", "883MBnccg0TLFV7MEZNVqmFRwmEiEBx0SbBiivVnEIefJgVkW4JTJQQJ99BBACHYHv6XJ3w3AAAAACOGmcwh"),
    api_version="2024-05-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://openaio3team64150034964.openai.azure.com/")
)

# Curation Style Enumeration
class CurationStyle(Enum):
    EMOTIONAL = "Emotional"
    INTERPRETIVE = "Interpretive"
    HISTORICAL = "Historical"
    CRITICAL = "Critical"
    NARRATIVE = "Narrative"
    Contemporary_Art_Critic = "Trend"
    Art_Appraiser = "Money"
    Aesthetic_Evaluation = "Praise"
    Image_Interpreter = "Blind"
    EDUCATIONAL = "Educational"

# Data Models
@dataclass
class ImageAnalysis:
    dense_caption: str
    tags: List[str]
    confidence_score: float

@dataclass
class ArtReference:
    title: str
    artist: str
    period: str
    medium: str
    description: str
    url: Optional[str] = None

@dataclass
class CurationRequest:
    user_prompt: str
    selected_style: CurationStyle
    image_analysis: ImageAnalysis
    reference_search: Optional[str] = None

@dataclass
class CurationResult:
    style: CurationStyle
    content: str
    references: List[ArtReference]
    metadata: Dict

# Unified Input Model
class UnifiedInput(BaseModel):
    query: Optional[str] = Field(default=None, description="Search query for artworks")
    has_images: Optional[bool] = Field(default=True, description="Filter for artworks with images")
    object_id: Optional[int] = Field(default=None, description="The ID of the artwork to get details for")
    artwork_title: Optional[str] = Field(default=None, description="Title of the artwork to search for")
    artist_name: Optional[str] = Field(default=None, description="Artist name to include in search")
    search_type: Optional[str] = Field(default="artwork", description="Type of search")

# Azure AI Assistant Integration
class AzureAIAssistant:
    def __init__(self,
                azure_endpoint="https://openaio3team64150034964.openai.azure.com/",
                api_key="883MBnccg0TLFV7MEZNVqmFRwmEiEBx0SbBiivVnEIefJgVkW4JTJQQJ99BBACHYHv6XJ3w3AAAAACOGmcwh",
                api_version="2024-05-01-preview"):
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version
        )
        self.logger = logging.getLogger(__name__)

        # Get existing assistant
        assistants = self.client.beta.assistants.list()
        self.assistant_id = assistants.data[0].id if assistants.data else None

    async def search_web(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        try:
            # Create thread
            thread = self.client.beta.threads.create()

            # Create search message
            message = self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=f"Search the web for: {query}\nProvide {max_results} most relevant results."
            )

            # Create run
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant_id
            )

            if self._wait_for_completion(thread.id, run.id):
                response = self._get_last_assistant_message(thread.id)
                return self._parse_search_results(response, max_results)

            return []

        except Exception as e:
            self.logger.error(f"Search error: {e}")
            return []

    def _wait_for_completion(self, thread_id: str, run_id: str, timeout: int = 30) -> bool:
        start_time = time.time()
        while True:
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )

            if run.status == "completed":
                return True
            elif run.status in ["failed", "cancelled"]:
                return False

            if time.time() - start_time > timeout:
                return False

            time.sleep(1)

    def _get_last_assistant_message(self, thread_id: str) -> Optional[str]:
        messages = self.client.beta.threads.messages.list(thread_id=thread_id)
        for msg in reversed(messages.data):
            if msg.role == "assistant":
                return msg.content[0].text.value
        return None

    def _parse_search_results(self, response: str, max_results: int) -> List[Dict[str, str]]:
        results = []
        if not response:
            return results

        try:
            lines = response.split('\n')
            current_result = {}

            for line in lines:
                line = line.strip()
                if line.startswith('Title:'):
                    if current_result and len(results) < max_results:
                        results.append(current_result.copy())
                    current_result = {'title': line[6:].strip()}
                elif line.startswith('Description:'):
                    current_result['snippet'] = line[12:].strip()

            if current_result and len(results) < max_results:
                results.append(current_result)

            return results[:max_results]

        except Exception as e:
            self.logger.error(f"Parse error: {e}")
            return []

# Updated MET Museum Tool Class
class MetMuseumTool(Tool):
    def __init__(self):
        super().__init__()
        self.base_url = "https://collectionapi.metmuseum.org/public/collection/v1"
        self.search_assistant = AzureAIAssistant()

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool의 실행 로직을 구현하는 메서드
        """
        if inputs.get("operation") == "search":
            return await self.search_artworks(UnifiedInput(**inputs))
        elif inputs.get("operation") == "details":
            return await self.get_artwork_details(UnifiedInput(**inputs))
        else:
            return {"error": True, "message": "Unknown operation"}

    def resources(self) -> List[Dict[str, Any]]:
        """
        Tool이 사용하는 리소스 정보를 반환하는 메서드
        """
        return [{
            "type": "api",
            "name": "met_api",
            "description": "Metropolitan Museum of Art Collection API",
            "url": self.base_url
        }]

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_artworks",
                    "description": "Search artworks in the Metropolitan Museum collection",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "has_images": {"type": "boolean", "description": "Filter for artworks with images"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_artwork_details",
                    "description": "Get detailed information about a specific artwork",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "object_id": {"type": "integer", "description": "Artwork ID"}
                        },
                        "required": ["object_id"]
                    }
                }
            }
        ]

    async def search_artworks(self, input: UnifiedInput) -> dict:
        try:
            # MET API search
            url = f"{self.base_url}/search"
            params = {"q": input.query, "hasImages": input.has_images}

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            met_result = response.json()

            # Additional context from Azure OpenAI
            web_results = await self.search_assistant.search_web(
                f"art history {input.query} analysis context"
            )

            return {
                "met_results": met_result,
                "web_context": web_results
            }

        except Exception as e:
            logger.error(f"Artwork search error: {e}")
            return {"error": True, "message": str(e)}

    async def get_artwork_details(self, input: UnifiedInput) -> dict:
        try:
            # Get MET API details
            url = f"{self.base_url}/objects/{input.object_id}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            artwork_data = response.json()

            # Get additional context
            web_results = await self.search_assistant.search_web(
                f"{artwork_data.get('title')} {artwork_data.get('artistDisplayName')} art analysis"
            )
            artwork_data['web_context'] = web_results

            # Add artwork analysis
            try:
                artwork_analysis = await self._analyze_artwork(artwork_data)
                artwork_data['analysis'] = artwork_analysis
            except Exception as e:
                logger.warning(f"Analysis failed: {e}")
                artwork_data['analysis'] = {"error": "Analysis failed"}

            return artwork_data

        except Exception as e:
            logger.error(f"Error getting artwork details: {e}")
            return {"error": True, "message": str(e)}

    async def _analyze_artwork(self, artwork_data: dict) -> dict:
        try:
            # 분석용 프롬프트 수정
            analysis_prompt = f"""
            다음 작품에 대한 예술사적 분석을 제공해주세요:
            제목: {artwork_data.get('title')}
            작가: {artwork_data.get('artistDisplayName')}
            시기: {artwork_data.get('period')}
            매체: {artwork_data.get('medium')}

            다음 측면에서 분석해주세요:
            1. 예술적 스타일과 특징
            2. 기술적 특성
            3. 역사적 맥락과 중요성
            """

            # 스레드 생성
            thread = self.search_assistant.client.beta.threads.create()

            # 메시지 생성 - chat.completions 사용
            completion = await self.search_assistant.client.chat.completions.create(
                model="gpt-4",  # 또는 귀하의 모델명
                messages=[
                    {"role": "user", "content": analysis_prompt}
                ]
            )

            analysis = completion.choices[0].message.content

            return {
                "style_analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"분석 오류: {e}")
            return {"error": "분석 실패"}

# Main execution
async def main():
    print("프로그램 시작...")  # 실행 시작 확인용

    try:
        # MET Tool 초기화
        met_tool = MetMuseumTool()
        print("MET Tool 초기화 완료")

        # 검색 실행
        search_params = {
            "operation": "search",
            "query": "Vincent van Gogh",
            "has_images": True
        }

        print("검색 실행 중...")
        search_results = await met_tool.execute(search_params)  # execute 메서드 사용

        # 검색 결과 처리
        if search_results.get("error"):
            print(f"검색 중 오류 발생: {search_results.get('message')}")
            return

        met_results = search_results.get("met_results", {})
        web_context = search_results.get("web_context", [])

        print("\n=== 검색 결과 ===")
        print(f"MET 컬렉션에서 {len(met_results.get('objectIDs', []))}개 작품 발견")

        print("\n=== 웹 컨텍스트 ===")
        for idx, result in enumerate(web_context, 1):
            print(f"\n{idx}. 제목: {result.get('title')}")
            print(f"   설명: {result.get('snippet')}")

        # 첫 번째 작품 상세 정보 가져오기
        if met_results.get("objectIDs"):
            details_params = {
                "operation": "details",
                "object_id": met_results["objectIDs"][0]
            }
            artwork_details = await met_tool.execute(details_params)

            if artwork_details.get("error"):
                print(f"작품 상세 정보 조회 중 오류 발생: {artwork_details.get('message')}")
                return

            print("\n=== 작품 상세 정보 ===")
            print(f"제목: {artwork_details.get('title')}")
            print(f"작가: {artwork_details.get('artistDisplayName')}")
            print(f"제작 연도: {artwork_details.get('objectDate')}")
            print(f"\n작품 분석:\n{artwork_details.get('analysis', {}).get('style_analysis', '분석 정보 없음')}")

            print("\n추가 컨텍스트:")
            for idx, result in enumerate(artwork_details.get('web_context', []), 1):
                print(f"\n{idx}. 출처: {result.get('title')}")
                print(f"   정보: {result.get('snippet')}")

    except Exception as e:
        logger.error(f"테스트 실패: {e}")
        print(f"실행 중 오류 발생: {e}")
        print(traceback.format_exc())  # 상세 에러 정보 출력

if __name__ == "__main__":
    asyncio.run(main())
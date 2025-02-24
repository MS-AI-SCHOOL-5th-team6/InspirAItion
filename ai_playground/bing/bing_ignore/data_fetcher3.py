import time
import logging
import os
from dotenv import load_dotenv
import requests
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from openai import AzureOpenAI
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.WARNING,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# 입력 검증을 위한 Pydantic 모델
class UnifiedInput(BaseModel):
    query: Optional[str] = Field(default=None, description="Search query for artworks")
    has_images: Optional[bool] = Field(default=True, description="Filter for artworks with images")
    object_id: Optional[int] = Field(default=None, description="The ID of the artwork to get details for")
    artwork_title: Optional[str] = Field(default=None, description="Title of the artwork to search for")
    artist_name: Optional[str] = Field(default=None, description="Artist name to include in search")
    search_type: Optional[str] = Field(default="artwork", description="Type of search")




class MetMuseumTool:
    def __init__(self):
        self.base_url = "https://collectionapi.metmuseum.org/public/collection/v1"

    def search_artworks(self, input: UnifiedInput) -> dict:
        """Search for artworks in the Metropolitan Museum collection"""
        try:
            # 검색어가 없으면 에러 반환
            if not input.query:
                return {
                    "error": True,
                    "message": "Search query is required"
                }

            url = f"{self.base_url}/search"
            # 한글 검색어도 처리할 수 있도록 인코딩
            params = {
                "q": input.query.strip(),
                "hasImages": input.has_images
            }
            logger.info(f"Searching artworks with query: {input.query}")

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            result = response.json()
            
            # 검색 결과가 없는 경우 처리
            if not result.get("objectIDs"):
                return {
                    "error": True,
                    "message": f"No artworks found for query: {input.query}"
                }
                
            logger.info(f"Search results: {len(result.get('objectIDs', []))} artworks found")
            return result

        except Exception as e:
            logger.error(f"Search error: {e}")
            return {
                "error": True,
                "message": f"Search failed: {str(e)}"
            }

class ConcreteBingSearchTool:
    def __init__(self):
        self.bing_search_endpoint = os.getenv("AZURE_BING_ENDPOINT", "").strip()
        self.bing_subscription_key = os.getenv("AZURE_BING_SUBSCRIPTION_KEY", "").strip()

    def search_artwork_context(self, input: UnifiedInput) -> Dict[str, Any]:
        """Search for additional context about artwork using Bing Search"""
        try:
            search_query = input.artwork_title
            if input.artist_name:
                search_query += f" {input.artist_name}"

            if input.search_type == "historical_context":
                search_query += " historical background context"

            headers = {
                'Ocp-Apim-Subscription-Key': self.bing_subscription_key,
                'Content-Type': 'application/json'
            }

            params = {
                'q': search_query,
                'count': 5,
                'mkt': 'en-US',
                'responseFilter': 'Webpages'
            }

            response = requests.get(
                f"{self.bing_search_endpoint}/v7.0/search",
                headers=headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()

            search_results = response.json()
            
            processed_results = {
                "query": params['q'],
                "sources": []
            }

            if 'webPages' in search_results and 'value' in search_results['webPages']:
                for page in search_results['webPages']['value']:
                    processed_results['sources'].append({
                        'title': page.get('name', ''),
                        'snippet': page.get('snippet', ''),
                        'url': page.get('url', '')
                    })

            return processed_results

        except Exception as e:
            logger.error(f"Bing search error: {e}")
            return {
                "error": True,
                "message": f"Search failed: {str(e)}"
            }


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
        
        self.met_tool = MetMuseumTool()
        self.bing_tool = ConcreteBingSearchTool()
        
        assistants = self.client.beta.assistants.list()
        self.assistant_id = assistants.data[0].id if assistants.data else None

    def search_artwork_info(self, artwork_name: str) -> Dict[str, Any]:
        """Search for artwork information using MET Museum API and Bing Search"""
        try:
            if not artwork_name:
                logger.error("Artwork name is required")
                return None

            # MET Museum 검색
            search_params = {
                "query": artwork_name,
                "has_images": True
            }
            
            logger.info(f"Searching for artwork: {artwork_name}")
            search_results = self.met_tool.search_artworks(UnifiedInput(**search_params))
            
            if search_results.get("error"):
                logger.error(f"MET Museum search error: {search_results.get('message')}")
                return None
                
            object_ids = search_results.get("objectIDs", [])
            if not object_ids:
                logger.error(f"No artworks found for: {artwork_name}")
                return None
                
            # 첫 번째 작품의 상세 정보 가져오기
            artwork_details = self.met_tool.get_artwork_details(UnifiedInput(object_id=object_ids[0]))
            
            if artwork_details.get("error"):
                logger.error(f"Failed to get artwork details: {artwork_details.get('message')}")
                return None
                
            # Bing Search로 추가 컨텍스트 검색
            bing_search_params = {
                "artwork_title": artwork_details.get("title", ""),
                "artist_name": artwork_details.get("artistDisplayName", ""),
                "search_type": "historical_context"
            }
            
            bing_results = self.bing_tool.search_artwork_context(UnifiedInput(**bing_search_params))
            
            return {
                "artwork_details": artwork_details,
                "additional_context": bing_results.get("sources", [])
            }
            
        except Exception as e:
            logger.error(f"Error during artwork search: {e}")
            return None

    def analyze_artwork(self, artwork_name: str, debug: bool = False) -> Optional[Dict[str, str]]:
        """Analyze artwork using the combined information from MET Museum and Bing Search"""
        try:
            if debug:
                logging.getLogger().setLevel(logging.INFO)
            
            logger.info(f"Starting analysis for artwork: {artwork_name}")
            artwork_info = self.search_artwork_info(artwork_name)
            
            if not artwork_info:
                logger.error("Failed to fetch artwork information")
                return None

            thread = self.client.beta.threads.create()
            prompt = self._create_analysis_prompt(artwork_info)
            
            message = self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=prompt
            )

            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant_id
            )

            if self.wait_for_run_completion(thread.id, run.id):
                analysis = self.get_last_assistant_message(thread.id)
                return {
                    "artwork": artwork_name,
                    "analysis": analysis,
                    "details": artwork_info
                }
            
            return None

        except Exception as e:
            if debug:
                logger.error(f"작품 분석 중 오류 발생: {e}")
            return None

def main():
    ai_assistant = AzureAIAssistant()
    # 디버그 모드를 True로 설정하여 자세한 로그 확인
    result = ai_assistant.analyze_artwork("starry night", debug=True)
    
    if result:
        print("\n🎨 예술 작품 분석 결과:")
        print(f"제목: {result['details']['artwork_details'].get('title')}")
        print(f"작가: {result['details']['artwork_details'].get('artistDisplayName')}")
        print("\n분석:")
        print(result['analysis'])
    else:
        print("분석 실패 - 작품을 찾을 수 없거나 분석 중 오류가 발생했습니다.")

if __name__ == "__main__":
    main()
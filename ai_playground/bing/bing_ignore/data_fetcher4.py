import time
import logging
import os
from dotenv import load_dotenv
import requests
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from openai import AzureOpenAI
from datetime import datetime

# Logging setup
logging.basicConfig(
    level=logging.WARNING,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Input validation Pydantic model
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
            if not input.query:
                return {
                    "error": True,
                    "message": "Search query is required"
                }

            url = f"{self.base_url}/search"
            params = {
                "q": input.query.strip(),
                "hasImages": input.has_images
            }
            logger.info(f"Searching artworks with query: {input.query}")

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            result = response.json()
            
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

    def get_artwork_details(self, input: UnifiedInput) -> dict:
        """Get detailed information about a specific artwork"""
        try:
            if not input.object_id:
                return {
                    "error": True,
                    "message": "Object ID is required"
                }

            url = f"{self.base_url}/objects/{input.object_id}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            return response.json()

        except Exception as e:
            logger.error(f"Error getting artwork details: {e}")
            return {
                "error": True,
                "message": f"Failed to get artwork details: {str(e)}"
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
        
        assistants = self.client.beta.assistants.list()
        self.assistant_id = assistants.data[0].id if assistants.data else None

    def get_additional_context(self, artwork_title: str, artist_name: str) -> Dict[str, Any]:
        """Get additional context about artwork using Azure OpenAI"""
        try:
            thread = self.client.beta.threads.create()
            
            prompt = f"""Please provide historical and artistic context for the artwork '{artwork_title}' 
            by {artist_name}. Include information about:
            1. The historical period and cultural context
            2. The artist's style and influences
            3. The significance of this artwork in art history
            Please provide factual, well-researched information."""

            message = self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=prompt
            )

            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant_id
            )

            # Wait for completion and get response
            if self.wait_for_run_completion(thread.id, run.id):
                context = self.get_last_assistant_message(thread.id)
                return {
                    "sources": [{
                        "title": "AI-Generated Art Historical Context",
                        "content": context
                    }]
                }
            return {"sources": []}

        except Exception as e:
            logger.error(f"Error getting additional context: {e}")
            return {"sources": []}

    def search_artwork_info(self, artwork_name: str) -> Dict[str, Any]:
        """Search for artwork information using MET Museum API and Azure OpenAI"""
        try:
            if not artwork_name:
                logger.error("Artwork name is required")
                return None

            # MET Museum search
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
                
            # Get details for the first artwork
            artwork_details = self.met_tool.get_artwork_details(UnifiedInput(object_id=object_ids[0]))
            
            if artwork_details.get("error"):
                logger.error(f"Failed to get artwork details: {artwork_details.get('message')}")
                return None
                
            # Get additional context using Azure OpenAI
            additional_context = self.get_additional_context(
                artwork_details.get("title", ""),
                artwork_details.get("artistDisplayName", "")
            )
            
            return {
                "artwork_details": artwork_details,
                "additional_context": additional_context.get("sources", [])
            }
            
        except Exception as e:
            logger.error(f"Error during artwork search: {e}")
            return None

    def wait_for_run_completion(self, thread_id: str, run_id: str, timeout: int = 30) -> bool:
        """Wait for the assistant's run to complete"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            run = self.client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
            if run.status == "completed":
                return True
            elif run.status in ["failed", "cancelled", "expired"]:
                return False
            time.sleep(1)
        return False

    def get_last_assistant_message(self, thread_id: str) -> str:
        """Get the last message from the assistant in a thread"""
        messages = self.client.beta.threads.messages.list(thread_id=thread_id)
        for message in messages.data:
            if message.role == "assistant":
                return message.content[0].text.value
        return ""

    def analyze_artwork(self, artwork_name: str, debug: bool = False) -> Optional[Dict[str, str]]:
        """Analyze artwork using the combined information from MET Museum and Azure OpenAI"""
        try:
            if debug:
                logging.getLogger().setLevel(logging.INFO)
            
            logger.info(f"Starting analysis for artwork: {artwork_name}")
            artwork_info = self.search_artwork_info(artwork_name)
            
            if not artwork_info:
                logger.error("Failed to fetch artwork information")
                return None

            thread = self.client.beta.threads.create()
            
            # Create analysis prompt using artwork information
            artwork_details = artwork_info["artwork_details"]
            additional_context = artwork_info["additional_context"]
            
            prompt = f"""Please analyze the artwork:
            Title: {artwork_details.get('title')}
            Artist: {artwork_details.get('artistDisplayName')}
            Date: {artwork_details.get('objectDate')}
            Medium: {artwork_details.get('medium')}
            
            Additional Context: {additional_context[0]['content'] if additional_context else 'No additional context available'}
            
            Please provide:
            1. A detailed analysis of the artwork's composition and style
            2. Its historical and cultural significance
            3. The artist's techniques and influences
            """

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
                logger.error(f"Error during artwork analysis: {e}")
            return None

def main():
    ai_assistant = AzureAIAssistant()
    # Set debug mode to True for detailed logging
    result = ai_assistant.analyze_artwork("starry night", debug=True)
    
    if result:
        print("\n🎨 Art Analysis Results:")
        print(f"Title: {result['details']['artwork_details'].get('title')}")
        print(f"Artist: {result['details']['artwork_details'].get('artistDisplayName')}")
        print("\nAnalysis:")
        print(result['analysis'])
    else:
        print("Analysis failed - Could not find artwork or an error occurred during analysis.")

if __name__ == "__main__":
    main()
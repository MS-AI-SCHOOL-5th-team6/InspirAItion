import time
from openai import AzureOpenAI
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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
        
        # 기존 어시스턴트 목록에서 첫 번째 어시스턴트 사용
        assistants = self.client.beta.assistants.list()
        self.assistant_id = assistants.data[0].id if assistants.data else None
        
        logger.info(f"사용할 어시스턴트 ID: {self.assistant_id}")

    def wait_for_run_completion(self, thread_id, run_id, max_wait_time=60):
        """
        런 완료를 기다리는 메서드
        
        Args:
            thread_id (str): 스레드 ID
            run_id (str): 런 ID
            max_wait_time (int): 최대 대기 시간 (초)
        
        Returns:
            bool: 런 완료 여부
        """
        start_time = time.time()
        
        while True:
            # 런 상태 확인
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )
            
            # 런 상태에 따른 처리
            if run.status == "completed":
                return True
            elif run.status in ["failed", "cancelled"]:
                logger.error(f"런 상태 오류: {run.status}")
                return False
            
            # 최대 대기 시간 초과 확인
            if time.time() - start_time > max_wait_time:
                logger.error("런 완료 대기 시간 초과")
                return False
            
            # 잠시 대기
            time.sleep(1)

    def get_last_assistant_message(self, thread_id):
        """
        스레드의 마지막 어시스턴트 메시지 가져오기
        
        Args:
            thread_id (str): 스레드 ID
        
        Returns:
            str: 어시스턴트 메시지 내용
        """
        messages = self.client.beta.threads.messages.list(thread_id=thread_id)
        
        for msg in reversed(messages.data):
            if msg.role == "assistant":
                return msg.content[0].text.value
        
        return None

    def test_agent(self, artwork_name):
        """
        에이전트 테스트 함수
        """
        try:
            # 스레드 생성
            thread = self.client.beta.threads.create()
            logger.info(f"새 스레드 생성: {thread.id}")

            # 메시지 생성
            message = self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=f"Provide a comprehensive analysis of the artwork '{artwork_name}'"
            )
            logger.info(f"스레드 메시지 생성: {message.id}")

            # 스레드 런 생성
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant_id
            )
            logger.info(f"스레드 런 생성: {run.id}")

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
            logger.error(f"에이전트 테스트 중 오류 발생: {e}")
            return None

# 사용 예시
def main():
    ai_assistant = AzureAIAssistant()
    result = ai_assistant.test_agent("Starry Night")
    
    if result:
        print("🎨 예술 작품 분석 결과:")
        print(result['analysis'])
    else:
        print("분석 실패")

if __name__ == "__main__":
    main()
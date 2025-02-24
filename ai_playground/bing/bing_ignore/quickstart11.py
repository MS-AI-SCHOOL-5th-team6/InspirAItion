import time
from openai import AzureOpenAI
import logging

# 로깅 설정 변경
logging.basicConfig(
    level=logging.WARNING,  # 경고 이상의 로그만 출력
    format='%(message)s'
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

# 사용 예시
def main():
    ai_assistant = AzureAIAssistant()
    
    # 일반 모드 (최소한의 출력)
    result = ai_assistant.test_agent("starry night")
    
    if result:
        print("\n🎨 예술 작품 분석 결과:")
        print(result['analysis'])
    else:
        print("분석 실패")

    # 디버그 모드 (상세 정보 출력)
    # result_debug = ai_assistant.test_agent("Mona Lisa", debug=True)

if __name__ == "__main__":
    main()
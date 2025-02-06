
# #### gpt-4o-mini + DALL-E 3 + 통합 코드 이미지 파일 Azure Blob Storage 저장

# %%
import os 
import re 
import time
import requests  # 이미지 다운로드를 위해 추가  
from dotenv import load_dotenv  
from openai import AzureOpenAI

from azure.storage.blob import BlobServiceClient, BlobClient
  
# .env 파일 로드 (gpt4o-mini용 환경 변수)  
load_dotenv("gpt4o-mini.env")  
  
# Azure OpenAI 환경 변수 설정  
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")  
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")  
  
if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:  
    raise ValueError("환경 변수가 올바르게 설정되지 않았습니다. .env 파일을 확인하세요.")  
  
# Azure OpenAI 클라이언트 초기화  
GPT_CLIENT = AzureOpenAI(  
    azure_endpoint=AZURE_OPENAI_ENDPOINT,  
    api_key=AZURE_OPENAI_API_KEY,  
    api_version=AZURE_OPENAI_API_VERSION  
)  
  
def generate_prompt_with_gpt4o(user_input):  
    """  
    GPT-4o-mini를 사용해 DALL-E 3 프롬프트 생성  
    """  
    try:  
        print("GPT-4o-mini를 사용해 프롬프트를 생성합니다...")  
  
        # Azure OpenAI GPT 모델 호출  
        assistant = GPT_CLIENT.beta.assistants.create(  
            model="gpt-4o-mini",  
            instructions="""  
            You are an expert in converting user's natural language descriptions into DALL-E image generation prompts.
            Please generate prompts according to the following guidelines:

            ##Main Guidelines

            1. Carefully analyze the user's description to identify key elements.
            2. Use clear and specific language to write the prompt.
            3. Include details such as the main subject, style, composition, color, and lighting of the image.
            4. Appro-priately utilize artistic references or cultural elements to enrich the prompt.
            5. Add instructions about image quality or resolution if necessary.
            6. Evaluate if the user's request might violate DALL-E's content policy. If there's a possibility of violation, include a message in the user's original language: "This content may be blocked by DALL-E. Please try a different approach." and explain why blocked.
            7. Always provide the prompt in English, regardless of the language used in the user's request.

            ##Prompt Structure

            - Specify the main subject first, then add details.
            - Use adjectives and adverbs effectively to convey the mood and style of the image.
            - Specify the composition or perspective of the image if needed.

            ##Precautions

            - Do not directly mention copyrighted characters or brands.
            - Avoid violent or inappropriate content.
            - Avoid overly complex or ambiguous descriptions, maintain clarity.
            - Avoid words related to violence, adult content, gore, politics, or drugs.
            - Do not use names of real people.
            - Avoid directly mentioning specific body parts.

            ##Using Alternative Expressions

            Consider DALL-E's strict content policy and use visual synonyms with similar meanings to prohibited words. Examples:

            - "shooting star" → "meteor" or "falling star"
            - "exploding" → "bursting" or "expanding"

            ##Example Prompt Format

            "[Style/mood] image of [main subject]. [Detailed description]. [Composition/perspective]. [Color/lighting information]." Follow these guidelines to convert the user's description into a DALL-E-appropriate prompt. The prompt should be creative yet easy for AI to understand. If there's a possibility of content policy violation, notify the user and suggest alternatives.
            """,  
            temperature=0.7  
        )  
  
        # 새 스레드 생성  
        thread = GPT_CLIENT.beta.threads.create()  
  
        # 사용자 입력 추가  
        GPT_CLIENT.beta.threads.messages.create(  
            thread_id=thread.id,  
            role="user",  
            content=user_input  
        )  
  
        # 실행 요청  
        try:  
            run = GPT_CLIENT.beta.threads.runs.create(  
                thread_id=thread.id,  
                assistant_id=assistant.id  
            )  
        except Exception as e:  
            print("run 객체 생성 중 예외 발생:", str(e))  
            return None  
  
        # run 객체가 생성되지 않았으면 종료  
        if not run:  
            print("GPT-4o-mini 실행 실패: run 객체가 생성되지 않았습니다.")  
            return None  
  
        # 실행 상태 확인 (반복적으로 상태 확인)  
        while run.status in ['queued', 'in_progress', 'cancelling']:  
            time.sleep(1)  # 1초 대기  
            run = GPT_CLIENT.beta.threads.runs.retrieve(  
                thread_id=thread.id,  
                run_id=run.id  
            )
            print(f"현재 실행 상태: {run.status}")  # 상태 출력  
  
        # 실행 완료 시 결과 확인  
        if run.status == 'completed':  
            print("GPT-4o-mini 호출 성공!")  
  
            # 메시지에서 assistant의 응답 가져오기  
            messages = GPT_CLIENT.beta.threads.messages.list(thread_id=thread.id)  
  
            # 디버깅: 반환된 메시지 구조 확인  
            print("반환된 메시지 구조:", messages)  
  
            for message in reversed(messages.data):  
                if message.role == 'assistant':  
                    # message.content의 데이터 타입 확인  
                    if isinstance(message.content, str):  # content가 문자열인 경우  
                        response = message.content.strip()  
                        print("생성된 프롬프트:", response)  
                        return response  
                    elif isinstance(message.content, list):  # content가 리스트인 경우  
                        # 리스트 내부의 TextContentBlock에서 텍스트 추출  
                        extracted_texts = [  
                            block.text.value for block in message.content  
                            if hasattr(block, 'text') and hasattr(block.text, 'value')  
                        ]  
                        response = " ".join(extracted_texts).strip()  
                        print("생성된 프롬프트 (리스트 처리):", response)  
                        return response  
                    else:  
                        print("message.content가 예상치 못한 형식입니다:", type(message.content))  
                        return None  
  
            # assistant 응답이 없는 경우  
            print("assistant의 응답을 찾을 수 없습니다.")  
            return None  
  
        # 실행 실패 시 처리  
        if run.status == 'failed':  
            print(f"GPT-4o-mini 실행 실패: 상태 - {run.status}")  
            print("run 객체의 속성:", dir(run))  # run 객체의 속성 출력  
            print("run 객체의 데이터:", run.__dict__)  # run 객체의 데이터 출력  
  
            # 실패 이유를 명확하게 출력  
            last_error = getattr(run, 'last_error', None)  
            if last_error:  
                print("실패 코드:", last_error.code)  
                print("실패 메시지:", last_error.message)  
            else:  
                print("실패 이유를 확인할 수 없습니다. run 객체를 더 조사해보세요.")  
  
            # 실패 상태에 따른 추가 처리  
            if last_error and last_error.code == 'rate_limit_exceeded':  
                print("\n원인: 요청량 제한 초과 (Rate Limit Exceeded)")  
                print("조치: 24시간 동안 대기하거나 Azure 포털에서 요청량 제한을 늘리세요.")  
            else:  
                print("\n다른 이유로 실패한 것 같습니다. API 설정이나 네트워크 상태를 확인하세요.")  
  
            return None  
  
    except Exception as e:  
        print("GPT-4o-mini 호출 중 예외 발생:", str(e))  
        return None  

# DALL-E 환경 변수 로드  
load_dotenv("dalle3.env")  
  
# DALL-E 클라이언트 초기화  
DALLE_CLIENT = AzureOpenAI(  
    azure_endpoint=os.environ["AZURE_DALLE_ENDPOINT"],  
    api_key=os.environ["AZURE_DALLE_API_KEY"],  
    api_version=os.environ["AZURE_DALLE_API_VERSION"]  
)  

def generate_image_with_dalle(prompt):  
    """DALL-E를 사용해 이미지를 생성하고 로컬에 저장"""  
    try:  
        print("DALL-E를 사용해 이미지를 생성합니다...")  
  
        # DALL-E API 호출  
        result = DALLE_CLIENT.images.generate(  
            model="dall-e-3",  # 사용할 DALL-E 모델  
            prompt=prompt,    # 생성할 이미지에 대한 프롬프트  
            n=1               # 생성할 이미지 개수  
        )  
  
        # 결과 처리  
        if result and result.data:  
            image_url = result.data[0].url  
            print("DALL-E 호출 성공! 생성된 이미지 URL:", image_url)  

            # 이미지 다운로드 및 저장
            save_image(image_url, prompt)

            return image_url  
        else:  
            print("DALL-E 호출 실패: 결과가 비어 있습니다.")  
            return None  
    except Exception as e:  
        print("DALL-E 호출 중 예외 발생:", str(e))  
        return None  

# Azure Blob Storage 환경 변수 로드
load_dotenv("azure_storage.env")

# Azure Blob Storage 연결 문자열
connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

# Blob Service Client 생성
blob_service_client = BlobServiceClient.from_connection_string(connect_str)

def save_image_to_blob_storage(image_url, prompt):
    """이미지를 다운로드하여 Azure Blob Storage에 저장"""
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()

        # 컨테이너 이름
        container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")

        # 파일명에서 특수문자 제거 및 최대 길이 제한
        sanitized_filename = re.sub(r'[<>:"/\\|?*]', '', prompt[:30]).strip()
        filename = f"{sanitized_filename}.png"

        # Blob Client 생성
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=filename)

        # 이미지 데이터 업로드
        blob_client.upload_blob(response.content)

        print(f"이미지가 Azure Blob Storage에 저장되었습니다: {filename}")

    except Exception as e:
        print(f"Azure Blob Storage에 이미지 저장 중 오류 발생: {e}")

def main():  
    """전체 워크플로 실행"""  
    user_input = input("이미지 생성 아이디어를 입력하세요: ").strip()  
    if not user_input:  
        print("입력값이 비어 있습니다. 유효한 입력값을 제공합니다.")  
        return  
  
    # Step 1: GPT-4o-mini를 사용해 프롬프트 생성  
    prompt = generate_prompt_with_gpt4o(user_input)  
    if not prompt:  
        print("프롬프트 생성 실패. 워크플로를 종료합니다.")  
        return  
  
    # Step 2: DALL-E를 사용해 이미지 생성  
    image_url = generate_image_with_dalle(prompt)  
    if not image_url:  
        print("이미지 생성 실패. 워크플로를 종료합니다.")  
        return  
    
    # Step 3: Azure Blob Storage에 이미지 저장
    if image_url:
        save_image_to_blob_storage(image_url, prompt)
  
    print("\n=== 최종 결과 ===")  
    print(f"생성된 이미지가 'images/' 폴더에 저장되었습니다.")  
  
  
if __name__ == "__main__":  
    main()



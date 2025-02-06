import re
import logging
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render, get_object_or_404, redirect
from azure.storage.blob import BlobServiceClient
from django.conf import settings
from openai import AzureOpenAI
import requests
import uuid
from datetime import datetime

from .forms import PostWithAIForm, PostEditForm
from .models import Post, AIGeneration

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('ai_generation.log'),
        logging.StreamHandler()
    ]
)

GPT_CLIENT = AzureOpenAI(
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_key=settings.AZURE_OPENAI_API_KEY,
    api_version=settings.AZURE_OPENAI_API_VERSION
)

DALLE_CLIENT = AzureOpenAI(
    azure_endpoint=settings.AZURE_DALLE_ENDPOINT,
    api_key=settings.AZURE_DALLE_API_KEY,
    api_version=settings.AZURE_DALLE_API_VERSION
)

def generate_prompt_with_gpt4o(user_input):
    """GPT-4o를 사용해 DALL-E 3 프롬프트 생성"""
    try:
        logging.info("GPT-4o를 사용해 프롬프트를 생성합니다...")

        response = GPT_CLIENT.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert in converting user's natural language descriptions into DALL-E image generation prompts.
                    Please generate prompts according to the following guidelines:

                    ## Main Guidelines
                    1. Carefully analyse the user's description to identify key elements.
                    2. Use clear and specific language to write the prompt.
                    3. Include details such as the main subject, style, composition, colour, and lighting.
                    4. Appropriately utilise artistic references or cultural elements.
                    5. Add instructions about image quality or resolution if necessary.
                    6. Evaluate content policy violations and notify if blocked.
                    7. Always provide the prompt in English.

                    ## Prompt Structure
                    - Specify the main subject first, then add details.
                    - Use adjectives and adverbs for mood and style.
                    - Specify composition or perspective if needed.

                    ## Precautions
                    - No copyrighted characters or brands
                    - No violent or inappropriate content
                    - Avoid complex or ambiguous descriptions
                    - No words related to violence, adult content, gore, politics, or drugs
                    - No names or real people
                    - No specific body parts

                    ## Format Example:
                    "[Style/mood] image of [main subject]. [Detailed description]. [Composition]. [Colour/lighting]."
                    """
                },
                {"role": "user", "content": user_input}
            ],
            temperature=0.7
        )

        if response.choices and len(response.choices) > 0:
            generated_prompt = response.choices[0].message.content.strip()
            logging.info(f"생성된 프롬프트: {generated_prompt}")
            return generated_prompt
        return None
    
    except Exception as e:
        logging.error(f"GPT-4o 호출 중 예외 발생: {str(e)}", exc_info=True)
        return None
    
def save_image_to_blob(image_url, prompt, user_id):
    """이미지를 Azure Blob Storage에 저장"""
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]

        sanitised_prompt = re.sub(r'[<>:"/\\|?*]', '', prompt[:20]).strip()
        filename = f"user_{user_id}_{timestamp}_{unique_id}_{sanitised_prompt}.png"

        blob_service_client = BlobServiceClient.from_connection_string(settings.AZURE_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(
            container=settings.CONTAINER_NAME,
            blob=filename
        )

        blob_client.upload_blob(response.content, overwrite=True)
        logging.info(f"이미지가 Blob Storage에 저장되었습니다: {filename}")
        return blob_client.url
    
    except Exception as e:
        logging.error(f"Blob Storage 저장 중 오류 발생: {str(e)}", exc_info=True)
        return None

def generate_image_with_dalle(prompt):
    """DALL-E를 사용해 이미지를 생성"""
    try:
        logging.info("DALL-E를 사용해 이미지를 생성합니다...")

        result = DALLE_CLIENT.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1
        )

        if result and result.data:
            image_url = result.data[0].url
            logging.info(f"DALL-E 호출 성공! 생성된 이미지 URL: {image_url}")
            return image_url
        return None
    
    except Exception as e:
        logging.error(f"DALL-E 호출 중 예외 발생: {str(e)}", exc_info=True)
        return None
    
@login_required
def generate_image(request):
    """이미지 생성 뷰"""
    if request.method != "POST":
        return JsonResponse({"error": "POST method required"}, status=405)
    
    prompt = request.POST.get("prompt", "").strip()
    if not prompt:
        return JsonResponse({"error": "프롬프트를 입력해주세요."}, status=400)

    try:
        generated_prompt = generate_prompt_with_gpt4o(prompt)
        if not generated_prompt:
            return JsonResponse({"error": "프롬프트 생성에 실패했습니다."}, status=500)
        
        image_url = generate_image_with_dalle(generated_prompt)
        if not image_url:
            return JsonResponse({"error": "이미지 생성에 실패했습니다."}, status=500)

        return JsonResponse({
            "image_url": image_url,
            "generated_prompt": generated_prompt
        })
    
    except Exception as e:
        logging.error(f"이미지 생성 중 오류 발생: {str(e)}", exc_info=True)
        return JsonResponse({"error": str(e)}, status=500)

@login_required
def index_ai(request: HttpRequest) -> HttpResponse:
    posts = Post.objects.filter(user=request.user).order_by('-date_posted')
    ai_images = AIGeneration.objects.filter(user=request.user).order_by('-created_at')[:5]
    return render(request, "app/index_ai.html", {
        "posts": posts,
        "ai_images": ai_images
    })


def post_detail(request: HttpRequest, pk: int) -> HttpResponse:
    post = get_object_or_404(Post, pk=pk)
    return render(request, "app/post_detail.html", {"post": post})


@login_required
def create_post(request: HttpRequest) -> HttpResponse:
    if request.method == "POST":
        form = PostWithAIForm(request.POST)
        if form.is_valid():
            post = form.save(commit=False)
            post.user = request.user

            generated_image_url = request.POST.get('generated_image_url')
            if generated_image_url:
                blob_url = save_image_to_blob(generated_image_url, form.cleaned_data['prompt'], request.user.id)
                if blob_url:
                    post.image = generated_image_url

                    AIGeneration.objects.create(
                        user=request.user,
                        prompt=form.cleaned_data['prompt'],
                        generated_prompt="",
                        image_url=blob_url
                    )

            post.save()
            form.save_m2m()
            return redirect("index_ai")
    else:
        form = PostWithAIForm()
    return render(request, "app/create_post.html", {"form": form})


@login_required
def edit_post(request: HttpRequest, pk: int) -> HttpResponse:
    """게시물 수정 (이미지 수정 불가)"""
    post = get_object_or_404(Post, pk=pk)
    if request.method == "POST":
        form = PostEditForm(request.POST, instance=post)
        if form.is_valid():
            post = form.save()
            return redirect("post_detail", pk=pk)
    else:
        form = PostEditForm(instance=post)
    return render(request, "app/edit_post.html", {"form": form, "post": post})


@login_required
def delete_post(request: HttpRequest, pk: int) -> HttpResponse:
    post = get_object_or_404(Post, pk=pk)
    if request.method == "POST":
        post.delete()
        return redirect("index_ai")
    return render(request, "app/post_detail.html", {"post": post})
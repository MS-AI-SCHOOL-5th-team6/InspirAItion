from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import json
from .genAI import gen_ai_image
from .models import ImageGeneration
from django.shortcuts import render
from django.urls import path
from . import views
import os
import time
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.http import JsonResponse
from openai import AzureOpenAI
from .models import AIImageGeneration

# by Jay---
def play(request):
    return render(request, "ai_playground/play.html")

# by SM ---
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
        print("GPT-4o를 사용해 프롬프트를 생성합니다...")

        response = GPT_CLIENT.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are an assistant that generates creative visual prompts for DALL-E.
                    Provide concise, descriptive prompts suitable for generating high-quality images."""
                },
                {"role": "user", "content": user_input}
            ],
            temperature=0.7
        )

        if response.choices and len(response.choices) > 0:
            generated_prompt = response.choices[0].message.content.strip()
            print("생성된 프롬프트:", generated_prompt)
            return generated_prompt
        return None
    
    except Exception as e:
        print("GPT-4o 호출 중 예외 발생:", str(e))
        return None
    
def generate_image_with_dalle(prompt):
    """DALL-E를 사용해 이미지를 생성"""
    try:
        print("DALL-E를 사용해 이미지를 생성합니다...")

        result = DALLE_CLIENT.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1
        )

        if result and result.data:
            return result.data[0].url
        return None
    
    except Exception as e:
        print("DALL-E 호출 중 예외 발생:", str(e))
        return None
    
@login_required
def generate_image(request):
    """이미지 생성 뷰"""
    if request.method == "POST":
        user_input = request.POST.get("prompt", "").strip()

        if not user_input:
            return JsonResponse({"error": "프롬프트를 입력해주세요."}, status=400)
        
        generated_prompt = generate_prompt_with_gpt4o(user_input)
        if not generated_prompt:
            return JsonResponse({"error": "프롬프트 생성에 실패했습니다."}, status=500)

        generated_prompt = generate_prompt_with_gpt4o(user_input)
        if not generated_prompt:
            return JsonResponse({"error": "프롬프트 생성에 실패했습니다."}, status=500)
        
        image_url = generate_image_with_dalle(generated_prompt)
        if not image_url:
            return JsonResponse({"error": "이미지 생성에 실패했습니다."}, status=500)

        AIImageGeneration.objects.create(
            user=request.user,
            prompt=user_input,
            generated_prompt=generated_prompt,
            image_url=image_url
        )

        return JsonResponse({
            "image_url": image_url,
            "generated_prompt": generated_prompt
        })
    
    return render(request, "ai_playground/generate_image.html")

@login_required
def image_history(request):
    """사용자의 이미지 생성 히스토리 보기"""
    images = AIImageGeneration.objects.filter(user=request.user)
    return render(request, "ai_playground/image_history.html", {"images": images})


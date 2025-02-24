import asyncio
import sys
import os
import pytest
from unittest.mock import AsyncMock, patch
from dataclasses import asdict
from datetime import datetime

# 모듈 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 필요한 클래스들 임포트 (ai_playground\bing\curation_generator.py)
from ai_playground.bing.curation_generator import (
    CurationStyle,
    ImageAnalysis,
    ArtReference,
    CurationRequest,
    CurationResult,
    EnhancedCurationService
)

@pytest.fixture
def mock_clients():
    return {
        'gpt_client': AsyncMock(),
        'vision_client': AsyncMock(),
        'met_client': AsyncMock(),
        'bing_client': AsyncMock()
    }

@pytest.fixture
def curation_service(mock_clients):
    return EnhancedCurationService(
        mock_clients['gpt_client'],
        mock_clients['vision_client'],
        mock_clients['met_client'],
        mock_clients['bing_client']
    )

@pytest.fixture
def sample_image_analysis():
    return ImageAnalysis(
        dense_caption="A beautiful sunset over mountains",
        tags=["sunset", "mountains", "nature"],
        confidence_score=0.95
    )

@pytest.fixture
def sample_curation_request(sample_image_analysis):
    return CurationRequest(
        user_prompt="이 풍경화의 감성적인 측면을 분석해주세요",
        selected_style=CurationStyle.EMOTIONAL,
        image_analysis=sample_image_analysis,
        reference_search="sunset landscape paintings"
    )

# 기본 초기화 테스트
def test_service_initialization(curation_service):
    assert curation_service.gpt_client is not None
    assert curation_service.vision_client is not None
    assert curation_service.met_client is not None
    assert curation_service.bing_client is not None
    assert curation_service.style_prompts is not None

# 스타일 프롬프트 로딩 테스트
def test_style_prompts_loading(curation_service):
    assert CurationStyle.EMOTIONAL in curation_service.style_prompts
    assert "prompt" in curation_service.style_prompts[CurationStyle.EMOTIONAL]
    assert "required_references" in curation_service.style_prompts[CurationStyle.EMOTIONAL]

# 데이터 통합 테스트
def test_integrate_data(curation_service, sample_curation_request):
    ref_artworks = [
        {
            "title": "테스트 작품",
            "artistDisplayName": "테스트 작가",
            "period": "현대",
            "medium": "oil on canvas",
            "description": "테스트 설명"
        }
    ]
    context_data = {
        "similar_emotional_works": {
            "sources": [
                {
                    "title": "관련 작품",
                    "snippet": "테스트 컨텍스트",
                    "url": "http://test.com"
                }
            ]
        }
    }

    integrated = curation_service.integrate_data(
        sample_curation_request,
        ref_artworks,
        context_data
    )

    assert integrated["user_prompt"] == sample_curation_request.user_prompt
    assert integrated["selected_style"] == sample_curation_request.selected_style
    assert "image_analysis" in integrated
    assert "ref_artworks" in integrated
    assert "context_data" in integrated

# 참조 정보 컴파일 테스트
def test_compile_references(curation_service):
    ref_artworks = [
        {
            "title": "테스트 작품",
            "artistDisplayName": "테스트 작가",
            "period": "현대",
            "medium": "oil on canvas",
            "description": "테스트 설명",
            "primaryImage": "http://test.com/image.jpg"
        }
    ]
    
    context_data = {
        "similar_emotional_works": {
            "sources": [
                {
                    "title": "관련 작품",
                    "snippet": "테스트 컨텍스트",
                    "url": "http://test.com"
                }
            ]
        }
    }

    references = curation_service._compile_references(ref_artworks, context_data)
    
    assert len(references) == 2
    assert isinstance(references[0], ArtReference)
    assert references[0].title == "테스트 작품"
    assert references[0].artist == "테스트 작가"

# 큐레이션 생성 통합 테스트
@pytest.mark.asyncio
async def test_generate_curation(curation_service, sample_curation_request, mock_clients):
    # Mock 응답 설정
    mock_clients['met_client'].search_artworks.return_value = {
        "objectIDs": [1, 2, 3]
    }
    mock_clients['met_client'].get_artwork_details.return_value = {
        "title": "테스트 작품",
        "artistDisplayName": "테스트 작가",
        "period": "현대",
        "medium": "oil on canvas",
        "description": "테스트 설명"
    }
    mock_clients['bing_client'].search_artwork_context.return_value = {
        "sources": [
            {
                "title": "관련 작품",
                "snippet": "테스트 컨텍스트",
                "url": "http://test.com"
            }
        ]
    }
    mock_clients['gpt_client'].generate_text.return_value = {
        "generated_text": "테스트 큐레이션 텍스트"
    }

    result = await curation_service.generate_curation(sample_curation_request)

    assert isinstance(result, CurationResult)
    assert result.style == sample_curation_request.selected_style
    assert result.content == "테스트 큐레이션 텍스트"
    assert len(result.references) > 0
    assert isinstance(result.metadata, dict)
    
    
# pytest 실행 명령어
# (venv) PS C:\Users\USER\Desktop\InspirAItion_0221> pytest ai_playground/bing/tests/test_curation_generator.py
from django import forms
from .models import Post

class PostWithAIForm(forms.ModelForm):
    prompt = forms.CharField(
        widget=forms.Textarea,
        required=False,
        help_text='AI 이미지 생성을 위한 프롬프트'
    )
    is_public = forms.BooleanField(
        required=False,
        initial=False,
        help_text='공개 갤러리에 공유하기'
    )

    class Meta:
        model = Post
        fields = ["title", "content", "is_public"]

class PostEditForm(forms.ModelForm):
    class Meta:
        model = Post
        fields = ["title", "content"]
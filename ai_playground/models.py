from django.db import models
from django.contrib.auth.models import User

class ImageGeneration(models.Model):
    user_prompt = models.TextField()
    ai_prompt = models.TextField()
    image_url = models.URLField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.user_prompt

class AIImageGeneration(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    prompt = models.TextField()
    generated_prompt = models.TextField()
    image_url = models.URLField(max_length=1000)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username}'s image - {self.created_at}"

    class Meta:
        ordering = ['-created_at']
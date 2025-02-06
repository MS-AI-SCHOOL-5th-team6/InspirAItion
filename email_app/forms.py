from django import forms
from .models import Email

class EmailForm(forms.ModelForm):
    class Meta:
        model = Email
        fields = ['recipient', 'subject', 'body']

    def clean_recipient(self):
        recipient = self.cleaned_data.get('recipient')
        if not recipient or '@' not in recipient:
            raise forms.ValidationError('Invalid email address')  # 이메일 주소 유효성 검사
        return recipient

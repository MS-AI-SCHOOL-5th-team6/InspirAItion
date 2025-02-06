from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from .models import Email
from .forms import EmailForm
from django.core.mail import send_mail
from django.contrib import messages

@login_required
def send_email(request):
    if request.method == 'POST':
        form = EmailForm(request.POST)
        if form.is_valid():
            email = form.save(commit=False)
            email.sender = request.user  # 발신자 설정
            email.save()
            try:
                send_mail(
                    email.subject,
                    email.body,
                    email.sender.email,
                    [email.recipient.email],
                    fail_silently=False,
                )
                messages.success(request, 'Email sent successfully!')
                return redirect('email_list')  # 이메일 발송 후 이메일 목록 페이지로 리디렉션
            except Exception as e:
                messages.error(request, f'Error sending email: {e}')
        else:
            messages.error(request, 'Form is not valid. Please check your inputs.')
    else:
        form = EmailForm()
    return render(request, 'email_app/send_email.html', {'form': form})

@login_required
def email_list(request):
    # 로그인한 사용자가 받은 이메일 목록과 보낸 이메일 목록을 가져오기
    received_emails = Email.objects.filter(recipient=request.user)
    sent_emails = Email.objects.filter(sender=request.user)
    # 받은 이메일 목록과 보낸 이메일 목록을 템플릿에 전달하여 렌더링
    return render(request, 'email_app/email_list.html', {
        'received_emails': received_emails,
        'sent_emails': sent_emails,
    })

@login_required
def email_detail(request, email_id):
    # 이메일 ID를 기반으로 이메일을 가져옴, 해당 이메일이 로그인한 사용자의 이메일인지 확인
    email = get_object_or_404(Email, id=email_id, recipient=request.user)
    # 이메일 상세 정보를 email_detail.html로 전달하여 렌더링
    return render(request, 'email_app/email_detail.html', {'email': email})

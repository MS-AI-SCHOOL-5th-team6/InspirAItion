# team6/views.py
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from accounts.views import signup, profile_update  # accounts 앱의 views 모듈에서 뷰 함수를 임포트
from app.views import index, post_detail, create_post, edit_post, delete_post  # app 앱의 views 모듈에서 뷰 함수를 임포트
from email_app.views import send_email, email_detail, email_list

# redirect 함수로 관리자 페이지 접근
@login_required
def custom_admin(request):
    return redirect('admin')  # 관리자 페이지로 리디렉션

def home(request):
    return render(request, 'team6/home.html')

def about(request):
    return render(request, 'team6/about.html')

def services(request):
    return render(request, 'team6/services.html') # html정리 전(main 에는 about으로 연결해둠)

def our_team(request):
    return render(request, 'team6/our_team.html') # html정리 전(main 에는 about으로 연결해둠)

def board(request):
    return render(request, 'team6/board.html') # 새 게시판 앱 생성 예정

def contact_us(request):
    return render(request, 'team6/contact_us.html') # email_app 연결 예정정

def ai_play(request):
    return render(request, 'team6/ai_play.html')  #html만 있고, 아직 기능 merge 전

def art_gal(request):
    return render(request, 'team6/artgal.html') #html만 있고, 아직 기능 merge 전 


def index_ai(request):
    return render(request, 'app/index_ai.html')

'''
def index(request):
    return render(request, 'app/index.html')
'''


from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from azure.storage.blob import BlobServiceClient
from django.conf import settings
from .forms import SignUpForm, ProfileUpdateForm
from .models import Profile


def signup(request):
    if request.method == "POST":
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            Profile.objects.create(
                user=user,
                nickname=form.cleaned_data.get('nickname')
            )
            raw_password = form.cleaned_data.get("password1")
            user = authenticate(username=user.username, password=raw_password)
            login(request, user)
            #return redirect("/app/")
            return redirect("home")  # ✅ 변경된 부분
    else:
        form = SignUpForm()
    return render(request, "accounts/signup.html", {"form": form})

@login_required
def profile_update(request):   
    if request.method == 'POST':
        form = ProfileUpdateForm(request.POST, request.FILES, instance=request.user.profile)
        if form.is_valid():
            profile = form.save(commit=False)

            if 'image_file' in request.FILES:
                file = request.FILES['image_file']
                file_extension = file.name.split('.')[-1]
                file_name = f"profile_{request.user.username}.{file_extension}"

                try:
                    blob_service_client = BlobServiceClient.from_connection_string(
                        settings.AZURE_CONNECTION_STRING
                    )
                    container_name = settings.CONTAINER_NAME
                    blob_client = blob_service_client.get_blob_client(
                        container=container_name, blob=file_name
                    )

                    blob_client.upload_blob(file, overwrite=True)                
                    profile.profile_image = blob_client.url
                    
                except Exception as e:
                    print("==== Blob 업로드 실패 ====")
                    print("에러:", str(e))
            
            profile.save()
            #return redirect('index')
            return redirect("profile")  # ✅ 변경된 부분
    else:
        form = ProfileUpdateForm(instance=request.user.profile)

    return render(request, 'accounts/profile_update.html', {'form': form})
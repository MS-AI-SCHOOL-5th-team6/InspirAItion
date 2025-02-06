async function handleImageGeneration(event) {
    event.preventDefault();

    const userInput = document.getElementById('userInput').value;
    const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;

    const response = await fetch('/ai/generate-image/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfToken,
        },
        body: JSON.stringify({ user_input: userInput }),
    });

    const data = await response.json();

    // 결과 출력
    document.getElementById('result').innerHTML = `
        <h2>결과</h2>
        <p>사용자 프롬프트: ${data.user_prompt}</p>
        <p>AI 프롬프트: ${data.ai_prompt}</p>
        <img src="${data.image_url}" alt="Generated Image" />
    `;
}

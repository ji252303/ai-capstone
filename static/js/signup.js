document.getElementById('signup-form').addEventListener('submit', function(event) {
    const name = document.getElementById('name').value;
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    const confirmPassword = document.getElementById('confirm-password').value;

    if (password !== confirmPassword) {
        alert('비밀번호가 일치하지 않습니다!');
        event.preventDefault();
    }

    const formData = {
        name: name,
        username: username,
        password: password
    };

    fetch('/signup', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('회원가입이 성공적으로 완료되었습니다!');
            window.location.href = '../templates/login.html'; // 로그인 페이지로 이동
        } else {
            alert(data.message); // 오류 메시지 표시
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
});

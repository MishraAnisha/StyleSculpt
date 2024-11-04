document.getElementById('upload-button').onclick = async function() {
    const fileInput = document.getElementById('file-input');
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select a file.");
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict', { // Using relative path since Flask serves it
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const result = await response.json();
        document.getElementById('result').innerText = 
            result.error ? result.error : `Class: ${result.class}, Index: ${result.index}`;
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('result').innerText = 'An error occurred during the upload.';
    }
};

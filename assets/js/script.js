function validateFile() {
        const fileInput = document.getElementById('audioFile');
        const file = fileInput.files[0];
        
        if (!file) {
            alert("Please select a file to upload.");
            return false;
        }

        const fileName = file.name;
        const fileExtension = fileName.split('.').pop().toLowerCase();

        if (fileExtension !== 'wav') {
            alert("Only WAV files are allowed. Please upload a valid WAV file.");
            return false;
        }

        // Validate the MIME type
        const validMimeTypes = ['audio/wav', 'audio/x-wav'];
        if (!validMimeTypes.includes(file.type)) {
            alert("The file you uploaded is not a valid WAV file. Please upload a valid WAV file.");
            return false;
        }

            // If validation passes, upload the file via AJAX
            uploadFile(file);
            return false; // Prevent default form submission
        }
        
            function uploadFile(file) {
                const formData = new FormData();
                formData.append('audioFile', file);
        
                fetch('/upload', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        alert(data.error);
                    } else {
                        alert(`Predicted Genre: ${data.genre}`);
                    }
                })
                .catch(error => {
                    alert('Error uploading file: ' + error);
                });
            }

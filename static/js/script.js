function validateFile() {
        const fileInput = document.getElementById('audioFile');
        const file = fileInput.files[0];
        
        if (!file) {
            alert("Please select a file to upload.");
            return false;
        }

        const fileName = file.name;
        const fileExtension = fileName.split('.').pop().toLowerCase();

        // if (fileExtension !== '.wav' || '.mp3') {
        //     alert("Only wav and mp3 files are allowed. Please upload a valid WAV file.");
        //     return false;
        // }

        //validate the MIME type
        //const validMimeTypes = ['audio/wav', 'audio/x-wav', 'audio/mp3'];
        //if (!validMimeTypes.includes(file.type)) {
        //    alert("The file you uploaded is not a valid file. Please upload a valid file.");
        //    return false;
        //}

            //if validation passes, upload the file via AJAX
            upload_file(file);
            return false; // Prevent default form submission
        }
        
            // function uploadFile(file) {
            //     const formData = new FormData();
            //     formData.append('audioFile', file);
        
            //     fetch('/uploads', {
            //         method: 'POST',
            //         body: formData
            //     })
            //     .then(response => response.json())
            //     .then(data => {
            //         if (data.error) {
            //             alert(data.error);
            //         } else {
            //             alert(`Predicted Genre: ${predicted_genre}`);
            //         }
            //     })
            //     .catch(error => {
            //         alert('Error uploading file: ' + error);
            //     });
            // }

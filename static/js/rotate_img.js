// document.addEventListener('DOMContentLoaded', function () {
//     const fileInput = document.getElementById('file-input');
//     const fileInputCarousel = document.getElementById('file-input-carousel');
//     const imageWrapper = document.getElementById('image-wrapper');
//     const carousel = document.getElementById('carousel');
//     const carouselImages = document.querySelector('.carousel-images');
//     const carouselProcessed = document.getElementById('carousel-processed');
//     const carouselInitial = document.getElementById('carousel-initial');
//     const dartboard = document.getElementById('dartboard');
//     let rotationInterval;
//     let currentRotation = 0;
//     const rotationSpeed = 9;  // Speed of rotation
//     const initialImageUrl = dartboard.src;  // Store the initial image URL

//     // Handle drag and drop for image wrapper
//     imageWrapper.addEventListener('dragover', (event) => {
//         event.preventDefault();
//         imageWrapper.classList.add('drag-over');
//     });

//     imageWrapper.addEventListener('dragleave', () => {
//         imageWrapper.classList.remove('drag-over');
//     });

//     imageWrapper.addEventListener('drop', (event) => {
//         event.preventDefault();
//         imageWrapper.classList.remove('drag-over');
//         const files = event.dataTransfer.files;
//         if (files.length > 0) {
//             fileInput.files = files;
//             resetToInitialImage();  // Reset to the initial image
//             startRotation();  // Start rotating immediately
//             uploadImage(files[0]);
//         }
//     });

//     // Handle clicking the image for file selection
//     imageWrapper.addEventListener('click', () => {
//         fileInput.click();
//     });

//     // Auto-submit form after selecting a file
//     fileInput.addEventListener('change', () => {
//         if (fileInput.files.length > 0) {
//             resetToInitialImage();  // Reset to the initial image
//             startRotation();  // Start rotating immediately
//             uploadImage(fileInput.files[0]);
//         }
//     });

//     // Handle drag and drop for carousel
//     carousel.addEventListener('dragover', (event) => {
//         event.preventDefault();
//         carousel.classList.add('drag-over');
//     });

//     carousel.addEventListener('dragleave', () => {
//         carousel.classList.remove('drag-over');
//     });

//     carousel.addEventListener('drop', (event) => {
//         event.preventDefault();
//         carousel.classList.remove('drag-over');
//         const files = event.dataTransfer.files;
//         if (files.length > 0) {
//             fileInputCarousel.files = files;
//             resetCarousel();  // Reset carousel
//             resetToInitialImage();  // Reset to initial image
//             startRotation();  // Start rotating immediately
//             uploadImage(files[0]);
//         }
//     });

//     // Start rotating the dartboard image
//     function startRotation() {
//         clearInterval(rotationInterval);  // Ensure no previous intervals are running
//         rotationInterval = setInterval(() => {
//             currentRotation += rotationSpeed;
//             if (currentRotation >= 360) {
//                 currentRotation -= 360;  // Reset the rotation counter after a full circle
//             }
//             dartboard.style.transition = 'transform 0.5s linear';  // Smooth transition
//             dartboard.style.transform = `rotate(${currentRotation}deg)`;
//         }, 500);
//     }

//     // Stop rotating the dartboard image
//     function stopRotation() {
//         clearInterval(rotationInterval);
//     }

//     // Reset the carousel to its initial state
//     function resetCarousel() {
//         carousel.style.display = 'none';  // Hide the carousel
//         carouselProcessed.src = '';  // Clear processed image
//         carouselInitial.src = '';  // Clear initial image
//         currentRotation = 0;  // Reset the rotation counter
//         carouselImages.style.transform = 'translateX(0)';  // Reset carousel position
//         showImageWrapper();  // Show the image wrapper again
//     }

//     // Reset the image to the initial state (title image) without stopping rotation
//     function resetToInitialImage() {
//         dartboard.src = initialImageUrl;  // Revert to the initial image
//         dartboard.style.transform = `rotate(${currentRotation}deg)`;  // Maintain current rotation state
//         carousel.style.display = 'none';  // Hide the carousel during the reset
//         showImageWrapper();  // Show the image wrapper again
//     }

//     // Show the image wrapper
//     function showImageWrapper() {
//         imageWrapper.style.display = 'flex';  // Show the image wrapper again
//     }

//     // Upload the image using AJAX
//     function uploadImage(file) {
//         const formData = new FormData();
//         formData.append('file', file);

//         const xhr = new XMLHttpRequest();
//         xhr.open('POST', '/game', true);

//         xhr.onload = function () {
//             if (xhr.status === 200) {
//                 const response = JSON.parse(xhr.responseText);
//                 dartboard.src = response.image_url;  // Update the image with the processed one

//                 // Immediately show the processed image and carousel without delay
//                 imageWrapper.style.display = 'none';  // Hide the image wrapper
//                 carouselProcessed.src = response.image_url;  // Set the processed image
//                 carouselInitial.src = URL.createObjectURL(file);  // Set the initial uploaded image
//                 carousel.style.display = 'flex';  // Show the carousel
//                 stopRotation();  // Stop rotation after the carousel is shown
//             } else {
//                 console.error('An error occurred while processing the image.');
//                 stopRotation();  // Stop rotating if there's an error
//             }
//         };

//         xhr.onerror = function () {
//             console.error('An error occurred while uploading the image.');
//             stopRotation();  // Stop rotating if there's an error
//         };

//         xhr.send(formData);
//     }

//     // Carousel controls
//     let currentSlide = 0;

//     function showSlide(index) {
//         const slides = document.querySelectorAll('.carousel-image');
//         if (index >= slides.length) {
//             currentSlide = 0;
//         } else if (index < 0) {
//             currentSlide = slides.length - 1;
//         } else {
//             currentSlide = index;
//         }
//         const offset = -currentSlide * 100;
//         document.querySelector('.carousel-images').style.transform = `translateX(${offset}%)`;
//     }

//     window.prevSlide = function () {
//         showSlide(currentSlide - 1);
//     };

//     window.nextSlide = function () {
//         showSlide(currentSlide + 1);
//     };
// });

document.addEventListener('DOMContentLoaded', function () {
    const fileInput = document.getElementById('file-input');
    const fileInputCarousel = document.getElementById('file-input-carousel');
    const imageWrapper = document.getElementById('image-wrapper');
    const carousel = document.getElementById('carousel');
    const carouselImages = document.querySelector('.carousel-images');
    const carouselProcessed = document.getElementById('carousel-processed');
    const carouselInitial = document.getElementById('carousel-initial');
    const dartboard = document.getElementById('dartboard');
    const prevButton = document.querySelector('.carousel-control.prev');
    const nextButton = document.querySelector('.carousel-control.next');
    let rotationInterval;
    let currentRotation = 0;
    const initialImageUrl = dartboard.src;  // Store the initial image URL

    // Handle drag and drop for image wrapper
    imageWrapper.addEventListener('dragover', (event) => {
        event.preventDefault();
        imageWrapper.classList.add('drag-over');
    });

    imageWrapper.addEventListener('dragleave', () => {
        imageWrapper.classList.remove('drag-over');
    });

    imageWrapper.addEventListener('drop', (event) => {
        event.preventDefault();
        imageWrapper.classList.remove('drag-over');
        const files = event.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            resetToInitialImage();  // Reset to the initial image
            startRotation();  // Start rotating immediately
            uploadImage(files[0]);
        }
    });

    // Handle clicking the image for file selection
    imageWrapper.addEventListener('click', () => {
        fileInput.click();
    });

    // Auto-submit form after selecting a file
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            resetToInitialImage();  // Reset to the initial image
            startRotation();  // Start rotating immediately
            uploadImage(fileInput.files[0]);
        }
    });

    // Handle drag and drop for carousel
    carousel.addEventListener('dragover', (event) => {
        event.preventDefault();
        carousel.classList.add('drag-over');
    });

    carousel.addEventListener('dragleave', () => {
        carousel.classList.remove('drag-over');
    });

    carousel.addEventListener('drop', (event) => {
        event.preventDefault();
        carousel.classList.remove('drag-over');
        const files = event.dataTransfer.files;
        if (files.length > 0) {
            fileInputCarousel.files = files;
            resetCarousel();  // Reset carousel
            resetToInitialImage();  // Reset to initial image
            startRotation();  // Start rotating immediately
            uploadImage(files[0]);
        }
    });

    // Start rotating the dartboard image
    function startRotation() {
        rotationInterval = setInterval(() => {
            currentRotation += 1;
            // if (currentRotation >= 360) {
            //     currentRotation -= 360;  // Reset the rotation counter after a full circle
            // }
            dartboard.style.transition = 'transform 0.1s linear';  // Smooth transition
            dartboard.style.transform = `rotate(${currentRotation}deg)`;
        }, 100); // Rotate every second
    }

    // Stop rotating the dartboard image
    function stopRotation() {
        clearInterval(rotationInterval);
        dartboard.style.transform = `rotate(${currentRotation}deg)`;
    }

    // Reset the carousel to its initial state
    function resetCarousel() {
        carousel.style.display = 'none';  // Hide the carousel
        carouselProcessed.src = '';  // Clear processed image
        carouselInitial.src = '';  // Clear initial image
        currentRotation = 0;  // Reset the rotation counter
        carouselImages.style.transform = 'translateX(0)';  // Reset carousel position
        hideArrows(); // Hide arrows when carousel is reset
        showImageWrapper();  // Show the image wrapper again
    }

    // Reset the image to the initial state (title image) without stopping rotation
    function resetToInitialImage() {
        dartboard.src = initialImageUrl;  // Revert to the initial image
        dartboard.style.transform = '';  // Reset rotation in case it was applied
        carousel.style.display = 'none';  // Hide the carousel during the reset
        hideArrows(); // Hide arrows when resetting to initial image
        showImageWrapper();  // Show the image wrapper again
        startRotation();  // Start rotating the initial image
    }

    // Show the image wrapper
    function showImageWrapper() {
        imageWrapper.style.display = 'flex';  // Show the image wrapper again
    }

    // Show arrows
    function showArrows() {
        prevButton.style.opacity = '1';
        prevButton.style.visibility = 'visible';
        nextButton.style.opacity = '1';
        nextButton.style.visibility = 'visible';
    }

    // Hide arrows
    function hideArrows() {
        prevButton.style.opacity = '0';
        prevButton.style.visibility = 'hidden';
        nextButton.style.opacity = '0';
        nextButton.style.visibility = 'hidden';
    }

    // Upload the image using AJAX
    function uploadImage(file) {
        const formData = new FormData();
        formData.append('file', file);

        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/game', true);

        xhr.onload = function () {
            if (xhr.status === 200) {
                const response = JSON.parse(xhr.responseText);
                dartboard.src = response.image_url;  // Update the image with the processed one

                // Delay after showing the processed image, then show the carousel
                setTimeout(() => {
                    imageWrapper.style.display = 'none';  // Hide the image wrapper
                    carouselProcessed.src = response.image_url;  // Set the processed image
                    carouselInitial.src = URL.createObjectURL(file);  // Set the initial uploaded image
                    carousel.style.display = 'flex';  // Show the carousel
                    showArrows();  // Show the arrows
                    stopRotation();  // Stop rotation after the delay
                }, 0);  // Delay after the processed image is shown
            } else {
                console.error('An error occurred while processing the image.');
                stopRotation();  // Stop rotating if there's an error
            }
        };

        xhr.onerror = function () {
            console.error('An error occurred while uploading the image.');
            stopRotation();  // Stop rotating if there's an error
        };

        xhr.send(formData);
    }

    // Carousel controls
    let currentSlide = 0;

    function showSlide(index) {
        const slides = document.querySelectorAll('.carousel-image');
        if (index >= slides.length) {
            currentSlide = 0;
        } else if (index < 0) {
            currentSlide = slides.length - 1;
        } else {
            currentSlide = index;
        }
        const offset = -currentSlide * 100;
        document.querySelector('.carousel-images').style.transform = `translateX(${offset}%)`;
    }

    window.prevSlide = function () {
        showSlide(currentSlide - 1);
    };

    window.nextSlide = function () {
        showSlide(currentSlide + 1);
    };
});
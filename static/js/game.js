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
    const notificationProcess = document.querySelector('.notification-process');
    const notificationError = document.querySelector('.notification-error');

    let rotationInterval;
    let currentRotation = 0;
    const initialImageUrl = dartboard.src;
    let isProcessing = false;

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
        if (isProcessing) {
            notificationProcess.classList.add('show');
            console.log('Processing image. Please wait.');
            setTimeout(() => {
                notificationProcess.classList.remove('show');
            }, 3000);
            return;
        }
        if (files.length > 0) {
            fileInput.files = files;
            resetToInitialImage();
            startRotation();
            uploadImage(files[0]);
        }
    });

    imageWrapper.addEventListener('click', () => {
        fileInput.click();
    });

    carousel.addEventListener('click', () => {
        fileInputCarousel.click();
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            resetToInitialImage();
            startRotation();
            uploadImage(fileInput.files[0]);
        }
    });

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
            resetCarousel();
            resetToInitialImage();
            startRotation();
            uploadImage(files[0]);
        }
    });

    function startRotation() {
        if (!rotationInterval) { // Ensure only one interval is created
            console.log("Starting rotation...");
            rotationInterval = setInterval(() => {
                currentRotation += 1;
                dartboard.style.transition = 'transform 0.1s linear'; // Smooth transition
                dartboard.style.transform = `rotate(${currentRotation}deg)`;
            }, 100); // Rotate every 100 milliseconds
        }
    }

    function stopRotation() {
        if (rotationInterval) {
            console.log("Stopping rotation..."); // Debugging log
            clearInterval(rotationInterval);
            rotationInterval = null; // Clear the interval reference
        }
        dartboard.style.transform = `rotate(${currentRotation}deg)`; // Keep the current rotation
    }

    function resetCarousel() {
        carousel.style.display = 'none';
        carouselProcessed.src = '';
        carouselInitial.src = '';
        currentRotation = 0;
        carouselImages.style.transform = 'translateX(0)';
        hideArrows();
        showImageWrapper();
    }

    function resetToInitialImage() {
        dartboard.src = initialImageUrl;
        dartboard.style.transform = '';
        carousel.style.display = 'none';
        hideArrows();
        showImageWrapper();
        startRotation();
    }

    function showImageWrapper() {
        imageWrapper.style.display = 'flex';
    }

    function showArrows() {
        prevButton.style.opacity = '1';
        prevButton.style.visibility = 'visible';
        nextButton.style.opacity = '1';
        nextButton.style.visibility = 'visible';
    }

    function hideArrows() {
        prevButton.style.opacity = '0';
        prevButton.style.visibility = 'hidden';
        nextButton.style.opacity = '0';
        nextButton.style.visibility = 'hidden';
    }

    function uploadImage(file) {
        isProcessing = true;
        const formData = new FormData();
        formData.append('file', file);
    
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/game', true);
    
        xhr.onload = function () {
            if (xhr.status === 200) {
                stopRotation();
                const response = JSON.parse(xhr.responseText);
                dartboard.src = response.image_url;    
                setTimeout(() => {
                    imageWrapper.style.display = 'none';  // Hide the image wrapper
                    carouselProcessed.src = response.image_url;  // Set the processed image
                    carouselInitial.src = URL.createObjectURL(file);  // Set the initial uploaded image
                    carousel.style.display = 'block';  // Show the carousel
                    showArrows();  // Show the arrows
                    isProcessing = false;  // Allow new uploads
                }, 0);
            } else {

                console.error('An error occurred while processing the image.');
                notificationError.classList.add('show');
                setTimeout(() => {
                    notificationError.classList.remove('show');
                }, 3000);
                stopRotation();
                isProcessing = false;
            }
        };
    
        xhr.onerror = function () {
            console.error('An error occurred while uploading the image.');
            notificationError.classList.add('show');
            setTimeout(() => {
                notificationError.classList.remove('show');
            }, 3000);
            stopRotation();
            isProcessing = false;
        };
    
    
        xhr.send(formData);
    }

    let currentSlide = 0;

    function showSlide(index) {
        const slides = document.querySelectorAll('.carousel-image');
        const totalSlides = slides.length;
    
        if (index >= totalSlides) {
            currentSlide = 0;
        } else if (index < 0) {
            currentSlide = totalSlides - 1;
        } else {
            currentSlide = index;
        }
    
        const offset = -currentSlide * 100;  // Calculate the offset
        document.querySelector('.carousel-images').style.transform = `translateX(${offset}%)`;
    }
    
    window.prevSlide = function () {
        showSlide(currentSlide - 1);
    };
    
    window.nextSlide = function () {
        showSlide(currentSlide + 1);
    };
    let startX = 0;
    let currentX = 0;
    let isSwiping = false;

    carousel.addEventListener('touchstart', (event) => {
        startX = event.touches[0].clientX;
        isSwiping = true;
    });

    carousel.addEventListener('touchmove', (event) => {
        if (!isSwiping) return;
        currentX = event.touches[0].clientX;
    });

    carousel.addEventListener('touchend', () => {
        const swipeDistance = startX - currentX;
        
        if (swipeDistance > 50) {
            // Swipe left: show the next slide
            nextSlide();
        } else if (swipeDistance < -50) {
            // Swipe right: show the previous slide
            prevSlide();
        }

        isSwiping = false;
    });
});
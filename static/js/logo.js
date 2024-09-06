$(document).ready(function() {
    const totalImages = 9;
    let currentImage = localStorage.getItem('currentLogo');

    if (!currentImage || currentImage >= totalImages) {
        currentImage = 1;
    } else {
        currentImage = parseInt(currentImage) + 1;
    }

    let logoSrc;

    if (window.location.pathname.includes('game')) {
        logoSrc = `../../static/images/logo/${currentImage}.jpg`;
    } else {
        logoSrc = `images/logo/${currentImage}.jpg`;
    }

    // Set the logo source
    $('#logo').attr('src', logoSrc);

    // Save the current logo image number to localStorage
    localStorage.setItem('currentLogo', currentImage);

    // Initially hide the logo and position it above the screen
    $('#logo').css({
        'opacity': '0',
        'transform': 'translateY(-100px)',  // Position above the view
        'transition': 'none'  // Disable transition initially to avoid immediate animation
    });

    // Trigger the animation after a short delay
    setTimeout(function() {
        $('#logo').css({
            'transition': 'transform 0.5s ease-out, opacity 1.3s ease-out',  // Add transition for both transform and opacity
            'transform': 'translateY(0)',  // Move down to its original position
            'opacity': '1'  // Fade in the opacity
        });
    }, 100);  // Slight delay to ensure the transition is visible

    // Handle when the user clicks on the logo
    $('#logo').on('click', function(event) {
        event.preventDefault();
        const link = $(this).closest('a').attr('href');  // Get the link from the logo's parent anchor

        // Trigger the logo fall
        $(this).css({
            'transform': 'translateY(100vh)',  // Move the logo off the bottom of the page
            'transition': 'transform 0.5s ease-out'  // Make the logo fall down
        });

        // Fade out the body
        $('body').addClass('fade-out');

        // Wait for the animations to complete, then navigate to the link
        setTimeout(function() {
            window.location.href = link;
        }, 420);  // Match this with the logo fall and fade-out duration
    });
});
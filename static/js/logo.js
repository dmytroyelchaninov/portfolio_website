$(document).ready(function() {
    const totalImages = 9;
    let currentImage = localStorage.getItem('currentLogo');

    if (!currentImage || currentImage >= totalImages) {
        currentImage = 1;
    } else {
        currentImage = parseInt(currentImage) + 1;
    }

    const logoSrc = `images/logo/${currentImage}.jpg`;
    $('#logo').attr('src', logoSrc);

    localStorage.setItem('currentLogo', currentImage);

    $('#logo').css({
        'opacity': '0',
        'transform': 'translateY(-100px)'
    });

    setTimeout(function() {
        $('#logo').css({
            'transition': 'transform 0.3s ease-out, opacity 0.5s ease-out',
            'transform': 'translateY(0)',
            'opacity': '1'
        });
    }, 300);
});
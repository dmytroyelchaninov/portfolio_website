const gulp = require('gulp');
const sass = require('gulp-sass')(require('sass'));
const pug = require('gulp-pug');
const browserSync = require('browser-sync').create();

// Task to compile SCSS to CSS
gulp.task('sass', function() {
    return gulp.src('src/scss/**/*.scss')  // Source of SCSS files
        .pipe(sass())  // Compiling SCSS to CSS
        .pipe(gulp.dest('dist/css'))  // Destination for CSS files
        .pipe(browserSync.stream());  // Inject changes without refreshing the page
});

// Task to compile Pug to HTML
gulp.task('pug', function() {
    return gulp.src('src/pug/**/*.pug')  // Source of Pug files
        .pipe(pug())  // Compiling Pug to HTML
        .pipe(gulp.dest('dist'))  // Destination for HTML files
        .pipe(browserSync.stream());  // Inject changes without refreshing the page
});

// Task to initialize BrowserSync and watch for file changes
gulp.task('serve', function() {
    browserSync.init({
        server: './dist'  // Base directory for the server
    });

    gulp.watch('src/scss/**/*.scss', gulp.series('sass'));  // Watch SCSS files
    gulp.watch('src/pug/**/*.pug', gulp.series('pug'));  // Watch Pug files
    gulp.watch('dist/*.html').on('change', browserSync.reload);  // Watch HTML files
});

// Default task
gulp.task('default', gulp.series('sass', 'pug', 'serve'));  // Run everything with 'gulp' command
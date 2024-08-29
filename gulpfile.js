const gulp = require('gulp');
const sass = require('gulp-sass')(require('sass'));
const pug = require('gulp-pug');
const browserSync = require('browser-sync').create();

// Task to compile SCSS to CSS
gulp.task('sass', function() {
    return gulp.src('src/scss/**/*.scss')
        .pipe(sass())
        .pipe(gulp.dest('static/css'))  // Compile CSS into the static directory
        .pipe(browserSync.stream());
});

// Task to compile Pug to HTML for static pages
gulp.task('pug-static', function() {
    return gulp.src(['src/pug/**/*.pug', '!src/pug/game.pug'])  // Exclude game.pug
        .pipe(pug())        
        .pipe(gulp.dest('static'))  // Compile static HTML into the dist directory
        .pipe(browserSync.stream());
});

// Task to compile Pug to HTML for Flask templates
gulp.task('pug-flask', function() {
    return gulp.src('src/pug/game.pug')  // Only process game.pug
        .pipe(pug())
        .pipe(gulp.dest('darts/templates'))  // Compile into the Flask templates directory
        .pipe(browserSync.stream());
});

// Task to initialize BrowserSync and watch for file changes
gulp.task('serve', function() {
    browserSync.init({
        server: 'static',  // Serve static files from dist directory
        port: 3000,
    });

    gulp.watch('src/scss/**/*.scss', gulp.series('sass'));  // Watch for changes in SCSS files
    gulp.watch('src/pug/**/*.pug', gulp.series('pug-static', 'pug-flask'));  // Watch for changes in Pug files
    gulp.watch('dist/*.html').on('change', browserSync.reload);  // Reload browser on static HTML changes
});

// Default task that runs on 'gulp' command
gulp.task('default', gulp.series('sass', 'pug-static', 'pug-flask', 'serve'));
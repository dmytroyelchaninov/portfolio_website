const gulp = require('gulp');
const sass = require('gulp-sass')(require('sass'));
const pug = require('gulp-pug');
const browserSync = require('browser-sync').create();

// Task to compile SCSS to CSS
function compileSass(done) {
    gulp.src('src/scss/**/*.scss')
        .pipe(sass().on('error', sass.logError))
        .pipe(gulp.dest('static/css'))  // Compile CSS into the static directory
        .pipe(browserSync.stream());
    done();  // Signal async completion
}

// Task to compile Pug to HTML for static pages
function compilePug(done) {
    gulp.src(['src/pug/**/*.pug', '!src/pug/game.pug'])  // Exclude game.pug from the general Pug task
        .pipe(pug({ pretty: false }))
        .pipe(gulp.dest('static'))  // Compile static HTML into the static directory
        .pipe(browserSync.stream());
    done();  // Signal async completion
}

// Task to compile game.pug separately into darts/templates
function compileGamePug(done) {
    gulp.src('src/pug/game.pug')
        .pipe(pug({ pretty: false }))
        .pipe(gulp.dest('darts/templates'))  // Compile game.pug into the darts/templates directory
        .pipe(browserSync.stream());
    done();  // Signal async completion
}

// Task to initialize BrowserSync and watch for file changes
function serve(done) {
    browserSync.init({
        server: {
            baseDir: 'static',  // Serve static files from the static directory
        },
        port: 3000,
    });

    gulp.watch('src/scss/**/*.scss', compileSass);  // Watch for changes in SCSS files
    gulp.watch(['src/pug/**/*.pug', '!src/pug/game.pug'], compilePug);  // Watch for changes in non-game Pug files
    gulp.watch('src/pug/game.pug', compileGamePug);  // Watch for changes in game.pug
    gulp.watch('static/*.html').on('change', browserSync.reload);  // Reload browser on static HTML changes
    done();  // Signal async completion
}

// Default task that runs on 'gulp' command
gulp.task('default', gulp.series(compileSass, compilePug, compileGamePug, serve));
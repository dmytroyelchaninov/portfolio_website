from flask import Flask, render_template

# Initialize the Flask application
app = Flask(__name__, 
            template_folder='templates',   # Specifies the directory for templates
            static_folder='../static')     # Specifies the directory for static files

@app.route('/game')
def game():
    # Render the game.html template when the /game route is accessed
    return render_template('game.html')

if __name__ == '__main__':
    # Run the Flask application in debug mode on port 5000
    app.run(debug=True, port=5000)
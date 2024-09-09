from flask import Flask, render_template, request, jsonify, url_for
import os
import random
import sys
from game.image_transform import process_image

script_dir = os.path.dirname(os.path.abspath(__file__))
yolos_dir = os.path.join(script_dir, 'yolos')
sys.path.append(yolos_dir)

app = Flask(__name__,
            template_folder='templates',
            static_folder='../static')

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/game', methods=['GET', 'POST'])
def game():

    message = random.choice(messages)

    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'message': 'No file'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'message': 'No selected file'}), 400
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process the image using the process_image function
            processed_image_path, score_list = process_image(filepath)
            score = ', '.join(map(str, score_list))

            # Construct the URL for the processed image
            processed_image_url = url_for('static', filename=f'uploads/{os.path.basename(processed_image_path)}')

            # Return a JSON response with the image URL, message, and score
            return jsonify({
                'image_url': processed_image_url,
                'message': message,
                'score': score
            })
    
    return render_template('game.html', message=message)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
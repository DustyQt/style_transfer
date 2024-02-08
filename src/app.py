from flask import Flask, request, render_template, flash, redirect
import os
from procesors.style_transfer import StyleTransferProcesor
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/app/files'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def style_transfer():
    return render_template('style_transfer.html')

@app.route('/', methods=['POST'])
def my_form_post():
    if 'content_image' not in request.files or 'style_image' not in request.files:
        app.logger.debug('incorrect_files')
        #return redirect(request.url)

    content_image = request.files['content_image']
    style_image = request.files['style_image']

    if content_image.filename == '' or style_image.filename == '':
        app.logger.debug('incorrect_files')
        return redirect(request.url)

    content_image_filename = secure_filename(content_image.filename)
    style_image_filename = secure_filename(style_image.filename)

    content_image.save(os.path.join(app.config['UPLOAD_FOLDER'], content_image_filename))
    style_image.save(os.path.join(app.config['UPLOAD_FOLDER'], style_image_filename))

    processor = StyleTransferProcesor(app.logger, UPLOAD_FOLDER)
    return processor.call_model(content_image_filename, style_image_filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
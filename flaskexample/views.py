import os
# We'll render HTML templates and access data sent by POST
# using the request object from flask. Redirect and url_for
# will be used to redirect the user once the upload is done
# and send_from_directory will help us to send/show on the
# browser the file that the user just uploaded
from flaskexample import app
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename
import hashlib
import helpers

# Initialize the Flask application
#app = Flask(__name__)

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = '/data/www/franklinfinder/flaskexample/uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set([ 'pdf', 'png', 'jpg', 'jpeg', 'gif','JPG'])

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


# Route that will process the file upload
@app.route('/upload', methods=['POST'])
def upload():
    # Get the name of the uploaded file
    file = request.files['file']
    # Check if the file is one of the allowed types/extensions
    print file.filename
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        fileext = filename.rsplit('.', 1)[1]
        m = hashlib.md5()
        m.update(filename+str(request.remote_addr))
        filehash = m.hexdigest()
        print filename,fileext,filehash

        # Move the file form the temporal folder to
        # the upload folder we setup
	#print os.getcwd()
	#print os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filehash+"."+fileext))
        showFileName = helpers.processImage(app.config['UPLOAD_FOLDER'],filehash+"."+fileext)
        # Redirect the user to the uploaded_file route, which
        # will basicaly show on the browser the uploaded file
        return render_template('result.html',
                               bounded_filename = url_for('uploaded_file', filename=showFileName)
        )
        #return redirect(url_for('uploaded_file', filename=showFileName))

# This route is expecting a parameter containing the name
# of a file. Then it will locate that file on the upload
# directory and show it on the browser, so if the user uploads
# an image, that image is going to be show after the upload
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)
#    return render_template('result.html',
#                           bounded_filename = "uploads/"+filename
#    )


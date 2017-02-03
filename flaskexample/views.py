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
from shutil import copyfile

# Initialize the Flask application
#app = Flask(__name__)

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = '/data/www/franklinfinder/flaskexample/uploads/'
app.config['PETFINDER_FOLDER'] = '/data/www/franklinfinder/flaskexample/petfinderImages/'
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


@app.route('/demo', methods=['POST'])
def demo():
        showFileName,idsToShow,dogData = helpers.processImage(app.config['UPLOAD_FOLDER'],'9e4a9ea04b896088eb32dfa77c25cf89.jpg')
        if len(idsToShow) == 0: print showFileName,idsToShow,dogData
        # Redirect the user to the uploaded_file route, which                                                                                                                                                
        # will basicaly show on the browser the uploaded file                                                                                                                                                
        if len(idsToShow) == 0:
            problem = "Sorry, we couldn\'t find a dog in your image."
            if showFileName != "":
                problem += "\nMaybe you are looking for a "+showFileName+"?"
            return render_template('problem.html',
                                   problem = problem,
                                   image = url_for('uploaded_file',filename=dogData)
            )
        basePetFinder="https://www.petfinder.com/petdetail/"
        return render_template('result.html',
                               bounded_filename = url_for('uploaded_file', filename=showFileName),
                               pf_img0 = url_for('petfinder_image',filename=idsToShow[0]+"_1.jpg"),
                               pf_img1 = url_for('petfinder_image',filename=idsToShow[1]+"_1.jpg"),
                               pf_img2 = url_for('petfinder_image',filename=idsToShow[2]+"_1.jpg"),
                               pf_href0 = basePetFinder+str(idsToShow[0]),
                               pf_href1 = basePetFinder+str(idsToShow[1]),
                               pf_href2 = basePetFinder+str(idsToShow[2]),
                               pf_name0 = dogData[0]["name"],
                               pf_name1 = dogData[1]["name"],
                               pf_name2 = dogData[2]["name"],
                               pf_desc0 = dogData[0]["desc"],
                               pf_desc1 = dogData[1]["desc"],
                               pf_desc2 = dogData[2]["desc"],
        )


# Route that will process the file upload
@app.route('/upload', methods=['POST'])
def upload():
    # Get the name of the uploaded file
    file1 = request.files['file']
    # Check if the file is one of the allowed types/extensions
    print file1.filename
    if file1 and allowed_file(file1.filename):
        # Make the filename safe, remove unsupported chars
        print type(file1)
        filename = secure_filename(file1.filename)
        fileext = filename.rsplit('.', 1)[1]
        m = hashlib.md5()
        m.update(filename+str(request.remote_addr))
        filehash = m.hexdigest()
        tmpfilename = os.path.join(app.config['UPLOAD_FOLDER'], filehash+"."+fileext)
        # Move the file form the temporal folder to
        # the upload folder we setup
	#print os.getcwd()
	#print os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file1.save(tmpfilename)
        m2 = hashlib.md5(open(tmpfilename, 'rb').read())
        hash2 = m2.hexdigest()
        ofname = hash2+"."+fileext
        copyfile(tmpfilename,os.path.join(app.config['UPLOAD_FOLDER'], ofname))
        showFileName,idsToShow,dogData = helpers.processImage(app.config['UPLOAD_FOLDER'],ofname)
        if len(idsToShow) == 0: print showFileName,idsToShow,dogData
        # Redirect the user to the uploaded_file route, which
        # will basicaly show on the browser the uploaded file
        if len(idsToShow) == 0:
            problem = "Sorry, we couldn\'t find a dog in your image."
            if showFileName != "":
                problem += "\nMaybe you are looking for a "+showFileName+"?"
            return render_template('problem.html',
                                   problem = problem,
                                   image = url_for('uploaded_file',filename=dogData)
            )
        basePetFinder="https://www.petfinder.com/petdetail/"
        return render_template('result.html',
                               bounded_filename = url_for('uploaded_file', filename=showFileName),
                               pf_img0 = url_for('petfinder_image',filename=idsToShow[0]+"_1.jpg"),
                               pf_img1 = url_for('petfinder_image',filename=idsToShow[1]+"_1.jpg"),
                               pf_img2 = url_for('petfinder_image',filename=idsToShow[2]+"_1.jpg"),
                               pf_href0 = basePetFinder+str(idsToShow[0]),
                               pf_href1 = basePetFinder+str(idsToShow[1]),
                               pf_href2 = basePetFinder+str(idsToShow[2]),
                               pf_name0 = dogData[0]["name"],
                               pf_name1 = dogData[1]["name"],
                               pf_name2 = dogData[2]["name"],
                               pf_desc0 = dogData[0]["desc"],
                               pf_desc1 = dogData[1]["desc"],
                               pf_desc2 = dogData[2]["desc"],
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
@app.route('/petfinderImages/<filename>')
def petfinder_image(filename):
    return send_from_directory(app.config['PETFINDER_FOLDER'],filename)

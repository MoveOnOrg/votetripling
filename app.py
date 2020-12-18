from flask import Flask, request, render_template
app = Flask(__name__)

@app.route('/')
def index():
    if request.method == 'POST':
        import pdb; pdb.set_trace();
        # handle file upload
        return
    else: 
        return render_template('upload_form.html')
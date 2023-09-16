from flask import Flask, render_template ,request,redirect, url_for
import os
app = Flask(__name__)
app.app_context().push()

#==============================Controllers=================================

@app.route('/',methods=['GET','POST'])
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
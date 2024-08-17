from flask import Flask, render_template
import pickle
app = Flask(__name__)
pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')


if __name__ =='main':
    app.run(debug=True)

from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///contact.db'
db = SQLAlchemy(app)

class Contact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    message = db.Column(db.String(200))
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        name = request.form['name']
        message = request.form['message']
        contact = Contact(name=name, message=message)
        db.session.add(contact)
        db.session.commit()
        return redirect('/')

    contacts = Contact.query.all()
    return render_template('index.html', contacts=contacts)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
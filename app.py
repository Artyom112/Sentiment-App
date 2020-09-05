from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import classifier

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sentiment.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

clf = classifier.Classifier()

class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    review = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(40), nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return '<Review %r>'% self.id


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        review = request.form['review']
        sentiment = clf.predict([review])

        obj = Review(review=review, sentiment=sentiment)

        try:
            db.session.add(obj)
            db.session.commit()
            return redirect('/')
        except:
            return 'Error while adding a review'
    else:
        #4
        reviews = Review.query.order_by(Review.date.desc()).all()
        return render_template('index.html', reviews=reviews)

@app.route('/<int:id>/delete')
def delete(id):
    rew = Review.query.get_or_404(id)
    try:
        db.session.delete(rew)
        db.session.commit()
        return redirect('/')
    except:
        return 'При удалении статьи произошла ощибка'

if __name__=="__main__":
    app.run(port=5001, debug=True)


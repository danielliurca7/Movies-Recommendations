from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import json

tf.get_logger().setLevel('ERROR')

model = tf.keras.models.load_model('./movie_model.h5')

movie_info = {}
with open('titles.csv', 'r') as info_file:
    first_line = True

    for line in info_file:
        if first_line:
            first_line = False
            continue


        line = line.split(',')

        link  = line[0]
        title = ','.join(line[1:-5])[2:-1]
        year  = line[-5]
        poster_link = ','.join(line[-4:])

        movie_info[link] = {'title': title, 'year': year, 'poster_link': poster_link}


app = Flask(__name__)


@app.route('/')
def login():
    return render_template('login.html')


@app.route('/rated_movies/<user>')
def rated_movies(user):
    with open('users_ratings.json', 'r') as f:
        user_data = f.read()
        users_ratings = json.loads(user_data)

    return render_template('movies.html', user_name=user, movie_info=movie_info, users_ratings=users_ratings)


@app.route('/recommended_movies/<user>')
def recommendations(user):
    with open('users_ratings.json', 'r') as f:
        user_data = f.read()
        users_ratings = json.loads(user_data)

    movies = []
    user_ratings = np.zeros(len(movie_info))

    for i, link in enumerate(movie_info):
        movies.append(link)

        if user in users_ratings and link in users_ratings[user]:
            user_ratings[i] = users_ratings[user][link] / 10

    prediction = [movies[index] for index in np.argsort(model.predict(user_ratings.reshape((1, -1))))[0][::-1] if user_ratings[index] == 0][:12]

    recommendations = {link: movie_info[link] for link in prediction}

    return render_template('recommendations.html', user_name=user, movie_info=recommendations)


@app.route('/rate', methods = ['POST'])
def rate():
    with open('users_ratings.json', 'w') as f:
        json.dump(json.loads(request.data), f, indent=4)

    return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 

if __name__ == '__main__':
    app.run(debug=False)
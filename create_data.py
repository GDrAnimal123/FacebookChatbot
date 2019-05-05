import pandas as pd
import re
import sqlite3
import pickle
import numpy as np


def write_to_files(x, y, x_path="", y_path=""):
    try:
        with open(x_path, 'w', encoding='utf8') as x_f, \
                open(y_path, 'w', encoding='utf8') as y_f:
            for x_cont, y_cont in zip(x, y):
                x_f.write(str(x_cont) + '\n')
                y_f.write(str(y_cont) + '\n')
    except IOError as e:
        print("Operation failed: {}".format(e.strerror))


def serialize(data, path):
    try:
        print("Serialize {} successfully".format(path))
        pickle.dump(data, open(path, "wb"))
    except IOError as e:
        print(e.strerror)


def deserialize(path):
    try:
        print("Deserialize {} successfully".format(path))
        return pickle.load(open(path, "rb"))
    except IOError as e:
        print(e.strerror)
        return None


def load_reddit_data(path, start="", end="", threshold_score=0, limit=100000):
    conn = sqlite3.connect('{}'.format(path))
    c = conn.cursor()
    sql_statement = "SELECT * FROM parent_reply WHERE parent NOT NULL and score > {} LIMIT {};".format(threshold_score, limit)
    df = pd.read_sql(sql_statement, conn)
    parent_vector = df["parent"]
    comment_vector = df["comment"]

    # Add start and end word for later encoder-decoder training
    comment_vector = comment_vector.apply(lambda x: start + " " + x.strip() + " " + end)

    x = parent_vector.values
    y = comment_vector.values

    return x, y


def load_rdany_data(path, start="", end=""):

    data = pd.read_csv(path)
    data = data[["source", "text"]]

    # Note chat messages has to follow Human -> Robot
    # order, if a human or robot makes response twice then
    # we remove the latter.
    # ** Can improve by concat all messages but too complicated
    # for a small dataset. :(

    # loc to access the group by row
    # then shift(1) to check the previous source
    data['source_next'] = data['source'].loc[data['source'].shift(1) != data["source"]]
    data = data.dropna()

    human_vector = data[(data["source"] == "human")]["text"]
    robot_vector = data[(data["source"] == "robot")]["text"]

    # Add start and end word for later encoder-decoder training
    robot_vector = robot_vector.apply(lambda x: start + " " + x.strip() + " " + end)

    # Return a batches of chat message between Human and Robot correspondingly.
    x = human_vector.values
    y = robot_vector.values

    return x, y

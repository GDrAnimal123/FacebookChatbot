import sqlite3
import json
from datetime import datetime
import time
import json

timeframe = '2015-05'
sql_transaction = []
# Specify the starting row that you want to continue process.
START_ROW = 18600000
CLEANUP_STEPS = 1000000
BUFFER_SIZE = 10000
# Minimum likes to select a comment.
THRESHOLD_SCORE = 2
# TABLE_NAME = "parent_reply"

connection = sqlite3.connect('data/{}.db'.format(timeframe))
c = connection.cursor()


def create_table():
    c.execute("CREATE TABLE IF NOT EXISTS parent_reply(parent_id TEXT PRIMARY KEY, comment_id TEXT UNIQUE, parent TEXT, comment TEXT, subreddit TEXT, unix INT, score INT)")


def format_data(data):
    data = data.replace('\n', ' newlinechar ').replace('\r', ' newlinechar ').replace('"', "'")
    data = data.encode('ascii', 'ignore').decode('ascii')
    return data


def transaction_bldr(sql):
    global sql_transaction
    sql_transaction.append(sql)
    if len(sql_transaction) > 1000:
        c.execute('BEGIN TRANSACTION')
        for s in sql_transaction:
            try:
                c.execute(s)
            except:
                pass
        connection.commit()
        sql_transaction = []


def sql_insert_replace_comment(commentid, parentid, parent, comment, subreddit, time, score):
    try:
        sql = """UPDATE parent_reply SET parent_id = ?, comment_id = ?, parent = ?, comment = ?, subreddit = ?, unix = ?, score = ? WHERE parent_id =?;""".format(parentid, commentid, parent, comment, subreddit, int(time), score, parentid)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion', str(e))


def sql_insert_has_parent(commentid, parentid, parent, comment, subreddit, time, score):
    try:
        sql = """INSERT INTO parent_reply (parent_id, comment_id, parent, comment, subreddit, unix, score) VALUES ("{}","{}","{}","{}","{}",{},{});""".format(parentid, commentid, parent, comment, subreddit, int(time), score)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion', str(e))


def sql_insert_no_parent(commentid, parentid, comment, subreddit, time, score):
    try:
        sql = """INSERT INTO parent_reply (parent_id, comment_id, comment, subreddit, unix, score) VALUES ("{}","{}","{}","{}",{},{});""".format(parentid, commentid, comment, subreddit, int(time), score)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion', str(e))


def acceptable(data):
    if len(data.split(' ')) > 1000 or len(data) < 1:
        return False
    elif len(data) > 32000:
        return False
    elif data == '[deleted]':
        return False
    elif data == '[removed]':
        return False
    else:
        return True


def find_parent(pid):
    try:
        sql = "SELECT comment FROM parent_reply WHERE comment_id = '{}' LIMIT 1".format(pid)
        c.execute(sql)
        result = c.fetchone()
        if result != None:
            return result[0]
        else:
            return False
    except Exception as e:
        # print(str(e))
        return False


def find_existing_score(pid):
    try:
        sql = "SELECT score FROM parent_reply WHERE parent_id = '{}' LIMIT 1".format(pid)
        c.execute(sql)
        result = c.fetchone()
        if result != None:
            return result[0]
        else:
            return False
    except Exception as e:
        # print(str(e))
        return False


def tuple_to_dict(tuple, keys=[]):
    json = {}
    assert len(tuple) == len(keys)
    if len(keys) == 0:
        keys = range(len(tuple))
    for i in range(len(tuple)):
        json[keys[i]] = tuple[i]
    return json


if __name__ == '__main__':
    create_table()
    row_counter = 0
    paired_rows = 0
    required_keys = ["id", "parent_id", "body", "score", "created_utc", "subreddit"]
    # with open('J:/chatdata/reddit_data/{}/RC_{}'.format(timeframe.split('-')[0],timeframe), buffering=1000) as f:
    conn = sqlite3.connect('data/RC_{}.sqlite'.format(timeframe))
    cursor = conn.cursor()
    # created_utc, subreddit uses to uniquely identify particular row (PK).
    sql = "select id, parent_id, body, score, created_utc, subreddit from May2015"
    entries_cursor = cursor.execute(sql)

    while True:
        buffers = entries_cursor.fetchmany(BUFFER_SIZE)
        if len(buffers) <= 0:
            break

        buffers_dicts_generator = (tuple_to_dict(x, required_keys) for x in buffers)

        for row in buffers_dicts_generator:
            # print(row)
            # time.sleep(555)
            row_counter += 1

            if row_counter > START_ROW:
                try:
                    parent_id = row['parent_id'].split('_')[1]
                    body = format_data(row['body'])
                    created_utc = row['created_utc']
                    score = row['score']

                    comment_id = row['id']

                    subreddit = row['subreddit']
                    parent_data = find_parent(parent_id)

                    existing_comment_score = find_existing_score(parent_id)
                    if existing_comment_score:
                        if score > existing_comment_score:
                            if acceptable(body):
                                sql_insert_replace_comment(comment_id, parent_id, parent_data, body, subreddit, created_utc, score)

                    else:
                        if acceptable(body):
                            if parent_data:
                                if score >= THRESHOLD_SCORE:
                                    sql_insert_has_parent(comment_id, parent_id, parent_data, body, subreddit, created_utc, score)
                                    paired_rows += 1
                            # else:
                            #     sql_insert_no_parent(comment_id, parent_id, body, subreddit, created_utc, score)
                except Exception as e:
                    print(str(e))

            if row_counter % 100000 == 0:
                # out of 2,200,000 only paired Rows = 195,826 (~ 9% -> 10% of data).
                print('Total Rows Read: {}, Paired Rows: {}, Time: {}'.format(row_counter, paired_rows, str(datetime.now())))

            if row_counter > START_ROW:
                if row_counter % CLEANUP_STEPS == 0:
                    print("Cleanin up!")
                    sql = "DELETE FROM parent_reply WHERE parent IS NULL"
                    c.execute(sql)
                    connection.commit()
                    c.execute("VACUUM")
                    connection.commit()

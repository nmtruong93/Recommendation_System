from django.db import connections
import pandas as pd

connection = connections['remote']


def get_data_by_pandas(query):
    df = pd.read_sql(sql=query, con=connection)
    return df


def get_data_by_cursor(query):
    try:
        cursor = connection.cursor()
        cursor.execute(query)
        record = cursor.fetchone()
        return record
    except Exception as error:
        print('Error while connecting to DB', error)

    return None


def get_all_data_by_cursor(query):
    try:
        cursor = connection.cursor()
        cursor.execute(query)
        records = cursor.fetchall()
        return records
    except Exception as error:
        print('Error while connecting to DB', error)

    return None

import sqlite3


def main() -> None:
    connection = sqlite3.connect("tutorial.db")
    cursor = connection.cursor()

    query = "CREATE TABLE movie(title, year, score)"
    result = cursor.execute(query)
    print(query)
    print(result.fetchone())
    print()

    query = "SELECT name FROM sqlite_master WHERE name='spam'"
    result = cursor.execute(query)
    print(query)
    print(result.fetchone() is None)
    print()

    query = "INSERT INTO movie VALUES ('Monty Python and the Holy Grail', 1975, 8.2), ('And Now for Something Completely Different', 1971, 7.5)"
    _ = cursor.execute(query)
    connection.commit()
    print(query)
    print("data inserted!")
    print()

    query = "SELECT score FROM movie"
    result = cursor.execute(query)
    print(result.fetchall())
    print()

    data = [
        ("Monty Python Live at the Hollywood Bowl", 1982, 7.9),
        ("Monty Python's The Meaning of Life", 1983, 7.5),
        ("Monty Python's Life of Brian", 1979, 8.0),
    ]
    query = "INSERT INTO movie VALUES(?, ?, ?)"
    cursor.executemany(query, data)
    connection.commit()
    print("data inserted!")
    print()

    query = "SELECT year, title FROM movie ORDER BY year"
    print(query)
    for row in cursor.execute(query):
        print(row)
    print()
    connection.close()

    print("DONE")


if __name__ == "__main__":
    main()

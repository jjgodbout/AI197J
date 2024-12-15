from connectors.snowflake_connector import SnowflakeConnection
from snowflake.snowpark.session import Session as SnowparkSession
from connectors.aws_connector import MySQLAuroraConnection
import time
from pymysql.err import OperationalError as PyMySQLOperationalError, InterfaceError as PyMySQLInterfaceError

def execute_sql(query, db_type, params=None, retries=2):
    while retries > 0:
        try:
            if db_type == "snowflake":
                sf_connection = SnowflakeConnection()
                conn = sf_connection.get_session()
                if not isinstance(conn, SnowparkSession):
                    raise TypeError("Snowflake connection not established properly")
                return conn.sql(query).collect()

            elif db_type == "watchtower":
                conn_instance = MySQLAuroraConnection()
                conn = conn_instance.get_connection()
                if not conn.open:
                    conn.ping(reconnect=True)
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    if query.strip().upper().startswith("SELECT"):
                        return cursor.fetchall()
                    else:
                        conn.commit()
                        return cursor.rowcount

            else:
                raise ValueError("Unsupported database type")

        except (PyMySQLOperationalError, PyMySQLInterfaceError) as e:
            print(f"Encountered an error: {e}")
            retries -= 1
            if retries <= 0:
                raise
            print(f"Retrying... {retries} attempts left")
            time.sleep(1)

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            print(f"Query: {query}")
            if params:
                print(f"Params: {params}")
            raise

    return None  # This line is reached if all retries are exhausted
import os
import pandas as pd
from sqlalchemy import create_engine, text
from multiprocessing import Pool
from datetime import datetime, timedelta

# --- Database Credentials (Adjust if necessary) ---
DB_HOST = '127.0.0.1'
DB_USER = 'root'
DB_PASSWORD = 'Hamilton1186!'
DB_NAME = 'test_weather_db' # Use a temporary test database name
DB_PORT = 3306

# --- Function to be run by each process ---
def test_db_connection(process_id):
    """
    Function executed by each process in the multiprocessing pool.
    It creates its own DB engine and performs a simple insert/read.
    """
    
    # CRITICAL: Create the SQLAlchemy engine *inside* the worker function.
    # Each process needs its own isolated connection.
    db_connection_str = f'mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    engine = None # Initialize engine to None
    
    try:
        engine = create_engine(db_connection_str)
        
        # Verify connection by attempting to get a connection from the pool
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1")).scalar()
            print(f"Process {process_id}: Successfully connected to DB. Test query result: {result}")

        # --- Create a dummy DataFrame ---
        test_date = datetime(2023, 1, 1) + timedelta(days=process_id)
        dummy_data = {
            'process_id': [process_id],
            'test_date': [test_date.strftime('%Y-%m-%d')],
            'temperature': [25.0 + process_id],
            'location': [f'TestCity_{process_id}']
        }
        df = pd.DataFrame(dummy_data)

        # --- Write DataFrame to a test table ---
        table_name = 'test_multiprocess_data'
        df.to_sql(name=table_name, con=engine, if_exists='append', index=False)
        print(f"Process {process_id}: Successfully wrote data for test_date {test_date.strftime('%Y-%m-%d')}.")

        # --- Read data back (optional verification) ---
        read_df = pd.read_sql(f"SELECT * FROM {table_name} WHERE process_id = {process_id}", con=engine)
        print(f"Process {process_id}: Read back {len(read_df)} rows for itself.")
        
        return f"Process {process_id} completed successfully."

    except Exception as e:
        print(f"Process {process_id}: ERROR - {e}")
        return f"Process {process_id} failed: {e}"
    finally:
        # Ensure the engine is disposed to close connections gracefully
        if engine:
            engine.dispose()
            print(f"Process {process_id}: Engine disposed.")

# --- Main execution block ---
if __name__ == '__main__':
    # --- Create the test database if it doesn't exist ---
    # This part connects directly without a pool to ensure the DB exists
    try:
        # Connect to MySQL server (without specifying a database initially)
        admin_engine = create_engine(f'mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/')
        with admin_engine.connect() as connection:
            # Check if the database exists
            result = connection.execute(text(f"SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = '{DB_NAME}'")).scalar()
            if not result:
                connection.execute(text(f"CREATE DATABASE {DB_NAME}"))
                print(f"Database '{DB_NAME}' created successfully.")
            else:
                print(f"Database '{DB_NAME}' already exists.")
            connection.commit() # Commit CREATE DATABASE
        admin_engine.dispose()
    except Exception as e:
        print(f"Error creating/checking database '{DB_NAME}': {e}")
        print("Please ensure your MySQL server is running and root user has privileges to create databases.")
        exit() # Exit if we can't even get the database set up

    print("\n--- Starting Multiprocessing Test ---")
    
    # Define the number of processes in the pool
    # Start with a small number (e.g., 2-4) to test
    num_processes_to_test = 4 
    
    # Create a list of arguments for each process (just an ID in this case)
    process_ids = list(range(num_processes_to_test))

    with Pool(num_processes_to_test) as pool:
        results = pool.map(test_db_connection, process_ids)

    print("\n--- Test Results Summary ---")
    for res in results:
        print(res)

    print("\n--- Final Verification (Optional) ---")
    try:
        final_engine = create_engine(f'mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
        with final_engine.connect() as connection:
            total_rows = connection.execute(text(f"SELECT COUNT(*) FROM test_multiprocess_data")).scalar()
            print(f"Total rows in 'test_multiprocess_data' table: {total_rows}")
            if total_rows == num_processes_to_test:
                print("All processes successfully inserted data!")
            else:
                print("Mismatch in row count. Some insertions might have failed.")
        final_engine.dispose()
    except Exception as e:
        print(f"Final verification failed: {e}")

    # Optional: Clean up the test table/database
    # You can manually delete the database from MySQL Workbench or run:
    # try:
    #     admin_engine = create_engine(f'mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/')
    #     with admin_engine.connect() as connection:
    #         connection.execute(text(f"DROP DATABASE {DB_NAME}"))
    #         connection.commit()
    #         print(f"Database '{DB_NAME}' dropped.")
    #     admin_engine.dispose()
    # except Exception as e:
    #     print(f"Error dropping database: {e}")
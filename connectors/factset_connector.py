from utils.secrets_retrieval import get_secret
import hashlib
import hmac
import requests
import json
from datetime import datetime
from utils.query_handler import execute_sql
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define your constants
key = os.getenv('factset_key')
keyId = os.getenv('factset_keyId')
counter = os.getenv('factset_counter')
username = os.getenv('factset_username')
serial = os.getenv('factset_serial')


# Function to increment counter
def increment_counter(current_counter):
    new_counter = int(current_counter) + 1
    return str(new_counter)

def get_current_counter():
    """
    Retrieves the current counter value from the Snowflake table.
    """
    query = "select max(counter_value) as counter_value from PROSPECTORDOCUMENTS.DOCUMENT_DOWNLOAD.FACTSET"  # Adjust the query as per your table schema
    result = execute_sql(query,'snowflake')
    return result[0][0] if result else None

def update_counter(new_counter):
    """
    Updates the counter value in the Snowflake table.
    """
    query = f"UPDATE PROSPECTORDOCUMENTS.DOCUMENT_DOWNLOAD.FACTSET SET counter_value = {new_counter}"  # Adjust the query as per your table schema
    execute_sql(query,'snowflake')

# Function to compute OTP
def compute_otp(key, counter):
    ba_key = bytearray.fromhex(key)
    my_int = int(counter).to_bytes(8, 'big', signed=True)
    my_hmac = hmac.new(ba_key, msg=my_int, digestmod=hashlib.sha512)
    digested_counter = my_hmac.digest()
    otp = digested_counter.hex()
    return otp

# Function to authenticate and retrieve session token
def authenticate(username, keyId, otp, serial):
    json_object = {
        'username': username,
        'keyId': keyId,
        'otp': otp,
        'serial': serial
    }

    OTP_url = 'https://auth.factset.com/fetchotpv1'
    payload = json.dumps(json_object)
    header = {'Content-Type': 'application/json'}

    r = requests.post(OTP_url, data=payload, headers=header)
    r_key = r.headers.get(key='X-DataDirect-Request-Key')
    r_token = r.headers.get(key='X-Fds-Auth-Token')

    return r_key, r_token

#Function called to authenicate with Factset
def authenticate_with_factset():
    # Get the current counter
    current_counter = get_current_counter()

    # Compute OTP
    otp = compute_otp(key, current_counter)

    # Authenticate and retrieve session token
    r_key, r_token = authenticate(username, keyId, otp, serial)

    # Increment the counter and update it in the database
    new_counter = increment_counter(current_counter)
    update_counter(new_counter)

    # Confirm authentication and session token work
    header = {'X-Fds-Auth-Token': r_token}
    Service_url = 'https://datadirect.factset.com/services/auth-test'

    r = requests.get(Service_url, headers=header)
    status_code = r.status_code
    status_code_reason = r.reason
    status_message = r.text

    return r_key, r_token, header,status_code, status_code_reason, status_message

import yaml
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader


# Read the config file
with open('config2.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Hash all passwords
hashed_passwords = stauth.Hasher.hash_passwords(config['credentials'])

# Update the config with hashed passwords
config['credentials'] = hashed_passwords

# Save the updated config
with open('config2.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)

print("Passwords have been hashed and saved to config.yaml")
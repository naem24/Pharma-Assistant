from cryptography.fernet import Fernet

# Decryption function
def decrypt_password() -> str:
    # Read the key 
    fernet_key = "6Qp9xMTRdbmFL0_P4ph4w4GME7ekO6Mm_cxEcX-y3B4="
    encrypted_password = "gAAAAABqXOJgQ4iKg3EkWWeYXKkXWdJoVi2dvCi4JccBP_0iVn9q2EWP6C-tJNze1vlBmZvCUAdZd72hF73MIG4tDoccSFQl2rlqGxS_WxSj6rILpU8tpDaN6-0KC3OUKarpe4Ie_-rL"
    fernet_key_utf = fernet_key.encode('utf-8')

    # Setup fernet with the key
    fernet = Fernet(fernet_key_utf)

    # Decrypt the password and return it for use
    return fernet.decrypt(encrypted_password.encode()).decode()
            
import bcrypt


def hash_password(password: str) -> str:
    """Hash a password using bcrypt and return as string."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    # Return as string for database storage
    return hashed.decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against a bcrypt hash."""
    try:
        # Encode both to bytes for comparison
        password_bytes = password.encode('utf-8')
        hash_bytes = hashed.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hash_bytes)
    except Exception as e:
        print(f"Password verification error: {e}")
        return False

from database import get_connection

# -------- FIXED ADMIN CREDENTIALS --------
ADMIN_EMAIL = "admin.com"
ADMIN_PASSWORD = "admin123"

def register_user(email, password):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (email, password, role) VALUES (?, ?, ?)",
            (email, password, "user")
        )
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()

def login_user(email, password):
    # ---- ADMIN LOGIN ----
    if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
        return ("admin",)

    # ---- USER LOGIN ----
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT role FROM users WHERE email=? AND password=?",
        (email, password)
    )
    user = cur.fetchone()
    conn.close()
    return user

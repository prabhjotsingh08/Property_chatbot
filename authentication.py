"""
authentication.py - Streamlit + Django Authentication Module
Handles user signup, login, and logout functionality
"""

import streamlit as st
import hashlib
import sqlite3
import re
from datetime import datetime, timedelta
import secrets

# -------------------------------
# Database Setup
# -------------------------------
def init_auth_db():
    """Initialize authentication database with users table."""
    conn = sqlite3.connect("auth.db", check_same_thread=False)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active INTEGER DEFAULT 1
        )
    """)
    
    # Create sessions table for token-based auth
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_token TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    
    conn.commit()
    return conn

# Initialize database
auth_conn = init_auth_db()

# -------------------------------
# Password Hashing (Django-style)
# -------------------------------
def hash_password(password, salt=None):
    """Hash password using SHA256 with salt (Django-style)."""
    if salt is None:
        salt = secrets.token_hex(16)
    
    # Combine salt and password
    pwd_hash = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"sha256${salt}${pwd_hash}"

def verify_password(password, hashed_password):
    """Verify password against hashed version."""
    try:
        algorithm, salt, pwd_hash = hashed_password.split('$')
        test_hash = hashlib.sha256((salt + password).encode()).hexdigest()
        return test_hash == pwd_hash
    except:
        return False

# -------------------------------
# Validation Functions
# -------------------------------
def validate_email(email):
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_username(username):
    """Validate username (alphanumeric, underscore, 3-20 chars)."""
    pattern = r'^[a-zA-Z0-9_]{3,20}$'
    return re.match(pattern, username) is not None

def validate_password(password):
    """
    Validate password strength:
    - At least 8 characters
    - Contains uppercase and lowercase
    - Contains at least one digit
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    
    if not re.search(r'\d', password):
        return False, "Password must contain at least one digit"
    
    return True, "Password is strong"

# -------------------------------
# User Authentication Functions
# -------------------------------
def create_user(username, email, password, full_name=""):
    """Create a new user account."""
    try:
        # Validate inputs
        if not validate_username(username):
            return False, "Username must be 3-20 characters (letters, numbers, underscore only)"
        
        if not validate_email(email):
            return False, "Invalid email format"
        
        is_valid, msg = validate_password(password)
        if not is_valid:
            return False, msg
        
        # Hash password
        password_hash = hash_password(password)
        
        # Insert user into database
        cursor = auth_conn.cursor()
        cursor.execute("""
            INSERT INTO users (username, email, password_hash, full_name)
            VALUES (?, ?, ?, ?)
        """, (username, email, password_hash, full_name))
        
        auth_conn.commit()
        return True, "Account created successfully!"
        
    except sqlite3.IntegrityError:
        return False, "Username or email already exists"
    except Exception as e:
        return False, f"Error creating account: {str(e)}"

def authenticate_user(username_or_email, password):
    """Authenticate user with username/email and password."""
    try:
        cursor = auth_conn.cursor()
        
        # Check if input is email or username
        if '@' in username_or_email:
            cursor.execute("""
                SELECT id, username, email, password_hash, full_name 
                FROM users 
                WHERE email = ? AND is_active = 1
            """, (username_or_email,))
        else:
            cursor.execute("""
                SELECT id, username, email, password_hash, full_name 
                FROM users 
                WHERE username = ? AND is_active = 1
            """, (username_or_email,))
        
        user = cursor.fetchone()
        
        if user and verify_password(password, user[3]):
            # Update last login
            cursor.execute("""
                UPDATE users 
                SET last_login = CURRENT_TIMESTAMP 
                WHERE id = ?
            """, (user[0],))
            auth_conn.commit()
            
            # Return user data
            return True, {
                "id": user[0],
                "username": user[1],
                "email": user[2],
                "full_name": user[4]
            }
        
        return False, "Invalid username/email or password"
        
    except Exception as e:
        return False, f"Authentication error: {str(e)}"

def create_session(user_id):
    """Create a session token for user."""
    try:
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(days=7)  # 7 days expiry
        
        cursor = auth_conn.cursor()
        cursor.execute("""
            INSERT INTO user_sessions (user_id, session_token, expires_at)
            VALUES (?, ?, ?)
        """, (user_id, session_token, expires_at))
        
        auth_conn.commit()
        return session_token
        
    except Exception as e:
        st.error(f"Session creation error: {e}")
        return None

def get_user_from_session(session_token):
    """Get user data from session token."""
    try:
        cursor = auth_conn.cursor()
        cursor.execute("""
            SELECT u.id, u.username, u.email, u.full_name
            FROM users u
            JOIN user_sessions s ON u.id = s.user_id
            WHERE s.session_token = ? 
            AND s.expires_at > CURRENT_TIMESTAMP
            AND u.is_active = 1
        """, (session_token,))
        
        user = cursor.fetchone()
        
        if user:
            return {
                "id": user[0],
                "username": user[1],
                "email": user[2],
                "full_name": user[3]
            }
        return None
        
    except Exception as e:
        st.error(f"Session validation error: {e}")
        return None

def logout_user(session_token):
    """Logout user by deleting session."""
    try:
        cursor = auth_conn.cursor()
        cursor.execute("DELETE FROM user_sessions WHERE session_token = ?", (session_token,))
        auth_conn.commit()
        return True
    except Exception as e:
        st.error(f"Logout error: {e}")
        return False

# -------------------------------
# Streamlit UI Components
# -------------------------------
def signup_form():
    """Display signup form."""
    st.markdown("### 📝 Create Account")
    
    with st.form("signup_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            full_name = st.text_input("Full Name", placeholder="John Doe")
            username = st.text_input("Username*", placeholder="johndoe")
        
        with col2:
            email = st.text_input("Email*", placeholder="john@example.com")
            password = st.text_input("Password*", type="password", placeholder="Min. 8 characters")
        
        confirm_password = st.text_input("Confirm Password*", type="password")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            submit = st.form_submit_button("Sign Up", use_container_width=True)
        with col2:
            if st.form_submit_button("Back to Login", use_container_width=True):
                st.session_state.auth_page = "login"
                st.rerun()
        
        if submit:
            # Validation
            if not username or not email or not password:
                st.error("Please fill in all required fields (*)")
            elif password != confirm_password:
                st.error("Passwords do not match!")
            else:
                success, message = create_user(username, email, password, full_name)
                
                if success:
                    st.success(message)
                    st.info("Please login with your credentials")
                    st.session_state.auth_page = "login"
                    st.rerun()
                else:
                    st.error(message)

def login_form():
    """Display login form."""
    st.markdown("### 🔐 Login to Your Account")
    
    with st.form("login_form"):
        username_or_email = st.text_input("Username or Email", placeholder="Enter username or email")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            submit = st.form_submit_button("Login", use_container_width=True)
        with col2:
            if st.form_submit_button("Sign Up", use_container_width=True):
                st.session_state.auth_page = "signup"
                st.rerun()
        
        if submit:
            if not username_or_email or not password:
                st.error("Please enter both username/email and password")
            else:
                success, result = authenticate_user(username_or_email, password)
                
                if success:
                    # Create session
                    session_token = create_session(result["id"])
                    
                    if session_token:
                        # Store in session state
                        st.session_state.authenticated = True
                        st.session_state.user = result
                        st.session_state.session_token = session_token
                        
                        st.success(f"Welcome back, {result['username']}!")
                        st.rerun()
                    else:
                        st.error("Failed to create session")
                else:
                    st.error(result)

def show_user_profile():
    """Display logged-in user profile in sidebar."""
    user = st.session_state.user
    
    st.sidebar.markdown("### 👤 User Profile")
    
    st.sidebar.markdown(f"""
    <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 1rem; border-radius: 8px; color: white; margin-bottom: 1rem;">
        <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">Welcome Back! 👋</div>
        <div style="font-size: 0.9rem; opacity: 0.9;">{user['full_name'] or user['username']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown(f"""
    <div style="background-color: white; padding: 0.8rem; border-radius: 8px; margin-bottom: 1rem; font-size: 0.85rem;">
        <div style="margin-bottom: 0.4rem;"><strong>Username:</strong><br>{user['username']}</div>
        <div><strong>Email:</strong><br>{user['email']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("🚪 Logout", use_container_width=True, type="primary"):
        logout_user(st.session_state.session_token)
        
        # Clear session state
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.session_token = None
        st.session_state.auth_page = "login"
        
        st.rerun()

# -------------------------------
# Main Authentication Handler
# -------------------------------
def authenticate():
    """
    Main authentication function to be called in Streamlit app.
    Returns True if user is authenticated, False otherwise.
    """
    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if "auth_page" not in st.session_state:
        st.session_state.auth_page = "login"
    
    if "user" not in st.session_state:
        st.session_state.user = None
    
    if "session_token" not in st.session_state:
        st.session_state.session_token = None
    
    # Check if user has valid session
    if st.session_state.session_token and not st.session_state.authenticated:
        user = get_user_from_session(st.session_state.session_token)
        if user:
            st.session_state.authenticated = True
            st.session_state.user = user
    
    # If authenticated, return True
    if st.session_state.authenticated:
        return True
    
    # Show authentication UI with custom styling
    st.markdown("""
    <style>
        /* Remove extra padding from main container during auth */
        section[data-testid="stSidebar"] {
            display: none;
        }
        .block-container {
            padding-top: 3rem !important;
            max-width: 600px !important;
            margin: 0 auto !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Centered title
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #1e3c72; margin: 0;">🏠 Smart Real Estate Assistant</h1>
        <p style="color: #666; margin-top: 0.5rem;">Please login to continue</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show appropriate form
    if st.session_state.auth_page == "login":
        login_form()
    elif st.session_state.auth_page == "signup":
        signup_form()
    
    return False

# -------------------------------
# Helper Function for Main App
# -------------------------------
def get_current_user():
    """Get currently logged-in user data."""
    return st.session_state.get("user", None)

def require_auth(func):
    """Decorator to require authentication for a function."""
    def wrapper(*args, **kwargs):
        if not st.session_state.get("authenticated", False):
            st.error("Please login to access this feature")
            authenticate()
            return None
        return func(*args, **kwargs)
    return wrapper
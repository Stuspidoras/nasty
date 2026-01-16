from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
import bcrypt
import jwt
import datetime
from functools import wraps
import redis
from config import Config

app = Flask(__name__)
CORS(app)

redis_client = redis.Redis(host=Config.REDIS_HOST, port=Config.REDIS_PORT, decode_responses=True)

def get_db_connection():
    conn = psycopg2.connect(
        host=Config.POSTGRES_HOST,
        port=Config.POSTGRES_PORT,
        database=Config.POSTGRES_DB,
        user=Config.POSTGRES_USER,
        password=Config.POSTGRES_PASSWORD
    )
    return conn

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')

        if not token:
            return jsonify({'error': 'Токен отсутствует'}), 401

        try:
            if token.startswith('Bearer '):
                token = token[7:]

            # Проверка в Redis
            if redis_client.get(f"blacklist:{token}"):
                return jsonify({'error': 'Токен недействителен'}), 401

            data = jwt.decode(token, Config.JWT_SECRET_KEY, algorithms=['HS256'])
            current_user_id = data['user_id']

        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Токен истёк'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Недействительный токен'}), 401

        return f(current_user_id, *args, **kwargs)

    return decorated

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()

    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({'error': 'Все поля обязательны'}), 400

    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s) RETURNING id",
            (username, email, password_hash)
        )
        user_id = cur.fetchone()[0]

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({
            'message': 'Пользователь успешно зарегистрирован',
            'user_id': user_id
        }), 201

    except psycopg2.IntegrityError:
        return jsonify({'error': 'Пользователь с таким email уже существует'}), 409
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()

    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'error': 'Email и пароль обязательны'}), 400

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("SELECT id, username, password_hash FROM users WHERE email = %s", (email,))
        user = cur.fetchone()

        cur.close()
        conn.close()

        if not user:
            return jsonify({'error': 'Неверный email или пароль'}), 401

        user_id, username, password_hash = user

        if not bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8')):
            return jsonify({'error': 'Неверный email или пароль'}), 401

        token = jwt.encode({
            'user_id': user_id,
            'username': username,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=Config.JWT_EXPIRATION_HOURS)
        }, Config.JWT_SECRET_KEY, algorithm='HS256')

        redis_client.setex(f"user:{user_id}:token",
                          Config.JWT_EXPIRATION_HOURS * 3600,
                          token)

        return jsonify({
            'message': 'Успешный вход',
            'token': token,
            'user': {
                'id': user_id,
                'username': username
            }
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/logout', methods=['POST'])
@token_required
def logout(current_user_id):
    token = request.headers.get('Authorization')[7:]

    # Добавление токена в черный список
    redis_client.setex(f"blacklist:{token}",
                      Config.JWT_EXPIRATION_HOURS * 3600,
                      'true')

    return jsonify({'message': 'Успешный выход'}), 200

@app.route('/api/auth/verify', methods=['GET'])
@token_required
def verify_token(current_user_id):
    """Проверка валидности токена"""
    return jsonify({
        'valid': True,
        'user_id': current_user_id
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
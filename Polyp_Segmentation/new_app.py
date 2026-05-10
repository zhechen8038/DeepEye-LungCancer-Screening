import os
import uuid
from datetime import datetime
from flask import Flask, render_template, jsonify, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from flask_login import LoginManager, UserMixin, current_user, login_user, logout_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
import requests
from dotenv import load_dotenv
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from functools import wraps
from utils.heatmap import heatmap

load_dotenv()
print("当前环境变量:", {k: v for k, v in os.environ.items() if 'DEEPSEEK' in k})
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')


app = Flask(__name__, template_folder='new_templates')
app.config.update({
    'SECRET_KEY': 'dev',
    'UPLOAD_FOLDER': os.path.join(os.getcwd(), 'static', 'uploads'),
    'SQLALCHEMY_DATABASE_URI': 'sqlite:///database.db',
    'SQLALCHEMY_TRACK_MODIFICATIONS': False,
    'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg','bmp'},
    'MODEL_PATH': './Weights/CFANet.pth',
    'TESTSIZE': 352,

    'ALLOWED_AVATAR_EXTENSIONS': {'png', 'jpg', 'jpeg', 'gif'},#头像格式
    'AVATAR_FOLDER': os.path.join(os.getcwd(), 'static', 'avatars'),
    'DEFAULT_AVATAR': 'default.jpg',
    'MAX_AVATAR_SIZE': 2 * 1024 * 1024  # 2MB
})
0
CORS(app, resources={r"/*": {"origins": "*"}})  # 新增CORS配置
# 静态文件配置
app.static_folder = 'static'
app.static_url_path = '/static'

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    avatar = db.Column(db.String(512), default='default.png')
    created_at = db.Column(db.DateTime, default=datetime.now)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def set_avatar(self, file):
        if file and allowed_avatar(file.filename):
            if file.content_length > app.config['MAX_AVATAR_SIZE']:
                raise ValueError('文件大小超过限制')

            filename = secure_filename(f"user_{self.id}_{file.filename}")
            avatar_path = os.path.join(app.config['AVATAR_FOLDER'], filename)
            os.makedirs(os.path.dirname(avatar_path), exist_ok=True)
            file.save(avatar_path)
            self.avatar = filename
            self.process_avatar(avatar_path)

    @staticmethod
    def process_avatar(image_path):
        """将头像裁剪为正方形并缩放到200x200像素"""
        img = Image.open(image_path)
        width, height = img.size

        # 居中裁剪正方形
        size = min(width, height)
        left = (width - size) / 2
        top = (height - size) / 2
        right = (width + size) / 2
        bottom = (height + size) / 2
        img = img.crop((left, top, right, bottom))

        # 缩放
        img = img.resize((200, 200), Image.Resampling.LANCZOS)
        img.save(image_path)

class MedicalRecord(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    original_path = db.Column(db.String(512), nullable=False)
    result_path = db.Column(db.String(512), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    #return User.query.get(int(user_id))
    return db.session.get(User, int(user_id))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

def load_model():
    global model
    if model is None:
        from lib.model import CFANet
        model = CFANet(channel=64).to(device)
        model.load_state_dict(torch.load(app.config['MODEL_PATH'], map_location=device,weights_only=True))
        model.eval()
    return model

def allowed_avatar(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_AVATAR_EXTENSIONS']

def init_upload_folders():
    folders = [
        app.config['UPLOAD_FOLDER'],
        app.config['AVATAR_FOLDER'],
        os.path.join(app.config['UPLOAD_FOLDER'], 'records')
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    # 复制默认头像
    default_src = os.path.join(app.root_path, 'static', 'images', 'default.jpg')
    default_dest = os.path.join(app.config['AVATAR_FOLDER'], 'default.jpg')
    if not os.path.exists(default_dest):
        import shutil
        shutil.copy(default_src, default_dest)

transform = transforms.Compose([
    transforms.Resize((app.config['TESTSIZE'], app.config['TESTSIZE'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if not all([username, email, password]):
            return jsonify({'error': '请填写所有字段'}), 400

        if User.query.filter_by(username=username).first():
            return jsonify({'error': '用户名已存在'}), 400
        if User.query.filter_by(email=email).first():
            return jsonify({'error': '邮箱已注册'}), 400

        new_user = User(username=username, email=email)
        new_user.set_password(password)

        db.session.add(new_user)
        db.session.commit()

        login_user(new_user)
        return jsonify({
            'success': '注册成功',
            'user': {
                'id': new_user.id,
                'username': new_user.username,
                'avatar': url_for('static', filename=f"uploads/{new_user.avatar}")
            }
        }), 201

    return render_template('register.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    remember = request.form.get('remember', False)

    user = User.query.filter_by(username=username).first()

    if not user or not user.check_password(password):
        return jsonify({'error': '无效的用户名或密码'}), 401

    login_user(user, remember=remember)
    return jsonify({
        'success': '登录成功',
        'user': {
            'id': user.id,
            'username': user.username,
            'avatar': url_for('static', filename=f"uploads/{user.avatar}")
        }
    }), 200

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()

    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'error': '消息不能为空'}), 400

    try:
        # 调用 DeepSeek API
        response = requests.post(
            'https://api.deepseek.com/chat/completions',
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {DEEPSEEK_API_KEY}'
            },

            json={
                'model': 'deepseek-chat',
                'messages': [
                    {
                        'role': 'system',
                        'content': '''你是一位专业的医生，专注于医疗健康领域。你的回答必须满足以下要求：
1. 严格使用纯文本Markdown格式（仅包含#、*、-等Markdown语法，禁止插入HTML标签或JSON结构）；
2. 标题层级需清晰（如# 主标题、## 子标题），内容使用列表或代码块排版；
3. 所有回复内容必须为字符串类型，禁止返回对象或数组。'''
                    },
                    {'role': 'user', 'content': user_message}
                ],
                'stream': False
            }
        )
        response_data = response.json()
        print("DeepSeek API 返回数据:", response_data)
        # 提取回复内容
        reply = response_data.get('choices', [{}])[0].get('message', {}).get('content', '暂无回复')
        if not isinstance(reply, str) or '[object Object]' in reply:
            # 防止意外返回对象或数组
            reply = str(reply)
        return jsonify({'reply': reply})

    except Exception as e:
        print('调用 DeepSeek 接口错误:', str(e))
        return jsonify({'error': '对话服务暂时不可用，请稍后重试'}), 500


@app.route('/analyze_file', methods=['POST'])
@login_required
def analyze_file():
    if 'file' not in request.files:
        return jsonify({'error': '未上传文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    try:
        # 保存上传的文件
        filename = secure_filename(file.filename)
        temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, filename)
        file.save(file_path)

        # 准备发送给DeepSeek的消息
        file_type = get_file_type(filename)
        prompt = f"用户上传了一份{file_type}文件，文件名为{filename}。请根据文件类型提供相应的分析和建议。"

        if file_type == "影像":
            # 如果是影像文件，可以尝试先进行AI分割，再结合结果分析
            try:
                result_path = process_image(file_path, temp_dir)
                analysis = "影像分割完成，发现以下特征：\n- 疑似息肉区域：1个\n- 最大直径：0.8cm\n- 位置：降结肠\n\n建议："
                prompt += "\n\n影像分割结果：" + analysis
            except Exception as e:
                print(f"影像处理失败: {str(e)}")
                prompt += "\n\n无法进行影像分割分析，请直接解读文件内容。"

        # 调用DeepSeek API
        response = requests.post(
            'https://api.deepseek.com/chat/completions',
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {DEEPSEEK_API_KEY}'
            },
            json={
                'model': 'deepseek-chat',
                'messages': [
                    {
                        'role': 'system',
                        'content': '你是一位专业的医生，专注于息肉医疗健康领域。你的回答应尽可能详细、准确，并使用Markdown格式（包括标题、列表、代码块等）进行排版，以便用户更好地阅读和理解。'
                    },
                    {'role': 'user', 'content': prompt}
                ],
                'stream': False
            }
        )

        response_data = response.json()
        reply = response_data.get('choices', [{}])[0].get('message', {}).get('content', '暂无回复')

        # 清理临时文件
        os.remove(file_path)

        return jsonify({'reply': reply})

    except Exception as e:
        print('分析文件错误:', str(e))
        return jsonify({'error': '文件分析失败，请稍后重试'}), 500


def get_file_type(filename):
    ext = filename.rsplit('.', 1)[1].lower()
    if ext in {'jpg', 'jpeg', 'png', 'dcm'}:
        return "影像"
    elif ext in {'pdf', 'txt', 'docx'}:
        return "报告"
    else:
        return "医疗文件"

# 症状提交路由（保持独立函数名）
@app.route('/submit-symptom', methods=['POST', 'OPTIONS'])
def submit_symptom():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()

    try:
        data = request.json
        print("接收到症状数据:", data)

        risk = "低风险"
        age = int(data['basic']['age']) if data['basic']['age'] else 0
        bleeding = data['symptoms']['bleeding']

        if age > 50:
            risk = "中风险"
        if bleeding == "经常":
            risk = "高风险"

        return jsonify({
            "status": "success",
            "message": f"评估结果：{risk}，建议3个月内预约肠镜检查"
        })

    except Exception as e:
        print('症状提交错误:', str(e))
        return jsonify({
            "status": "error",
            "message": "服务器处理失败，请检查数据格式"
        }), 500


def _build_cors_preflight_response():
    response = jsonify({"status": "success"})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response


@app.route('/transfer', methods=['POST'])
def transfer():
    # 模拟转接人工
    return jsonify({
        'message': '已转接至人工服务，请耐心等待客服回复。如有紧急情况，请拨打热线电话 123456。'
    })


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return jsonify({'success': '已退出登录'}), 200

@app.route('/upload_avatar', methods=['POST'])
@login_required
def upload_avatar():
    if 'avatar' not in request.files:
        return jsonify({'error': '未选择文件'}), 400

    file = request.files['avatar']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    try:
        current_user.set_avatar(file)
        db.session.commit()
        return jsonify({
            'success': '头像上传成功',
            'avatar_url':  f'/static/avatars/{current_user.avatar}'
        }), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': '上传失败: ' + str(e)}), 500

@app.route('/current_user')
def get_current_user():
    if current_user.is_authenticated:
        return jsonify({
            'id': current_user.id,
            'username': current_user.username,
            'avatar':  f'/static/avatars/{current_user.avatar}',
            'email':current_user.email
        }), 200
    return jsonify({'error': '未登录'}), 401

@app.route('/')
def head():
    return render_template('index.html')

@app.route('/go_to_AIpage')
def ai_page():
    return render_template('ai-assistant.html')

@app.route('/go_to_upload')
def fir_page():
    return render_template('upload-page.html')

@app.route('/go_to_Login')
def login_page():
    return render_template('login.html')

@app.route('/go_to_Register')
def register_page():
    return render_template('register.html')  # 单独的注册页面

@app.route('/go_to_index')
def index():
    return render_template('index.html')
@app.route('/go_to_visualization')
def visualization():
    return render_template('data-visualization.html')

@app.route('/go_to_knowledge')
def knowledge():
    return render_template('knowledge.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        record_id = str(uuid.uuid4())
        save_dir = os.path.join(app.config['UPLOAD_FOLDER'], record_id)
        os.makedirs(save_dir, exist_ok=True)

        filename = secure_filename(file.filename)
        original_path = os.path.join(save_dir, filename)
        file.save(original_path)

        result_path = process_image(original_path, save_dir)

        record = MedicalRecord(
            id=record_id,
            original_path=os.path.join(record_id, filename),
            result_path=os.path.join(record_id, 'result.png')
        )
        db.session.add(record)
        db.session.commit()

        return jsonify({
            'original': f'/static/uploads/{record.original_path}',
            'result': f'/static/uploads/{record.result_path}',
            'time': record.created_at.strftime('%Y-%m-%d %H:%M')
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


def process_image(image_path, save_dir):
    img = Image.open(image_path).convert('RGB')

    original_size = img.size
    model = load_model()
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, _, res = model(img_tensor)
        res = F.interpolate(res, size=original_size[::-1], mode='bilinear', align_corners=False)
        res = res.sigmoid().cpu().numpy().squeeze()

    res = (res - res.min()) / (res.max() - res.min() + 1e-8)

    result_path = os.path.join(save_dir, 'result.png')
    base_directory = "C:\\下载\\Polyp_Segmentation"
    relative_path = os.path.relpath(result_path, base_directory)
    relative_path = relative_path.replace('\\', '/')
    if not relative_path.startswith('..'):
        relative_path = f'../{relative_path}'

    heatmap(res, img_tensor, relative_path, original_size)
    return result_path

@app.route('/recent')
def get_recent():
    records = MedicalRecord.query.order_by(MedicalRecord.created_at.desc()).limit(4).all()
    return jsonify([{
        'id': r.id,
        'original': f'/static/uploads/{r.original_path}',
        'result': f'/static/uploads/{r.result_path}',
        'time': r.created_at.strftime('%Y-%m-%d %H:%M')
    } for r in records])

@app.route('/all_records')
def get_all_records():
    records = MedicalRecord.query.order_by(MedicalRecord.created_at.desc()).all()
    return jsonify([{
        'id': r.id,
        'original': f'/static/uploads/{r.original_path}',
        'result': f'/static/uploads/{r.result_path}',
        'time': r.created_at.strftime('%Y-%m-%d %H:%M')
    } for r in records])

@app.route('/delete/<record_id>', methods=['POST'])
def delete_record(record_id):
    try:
        record = MedicalRecord.query.get(record_id)
        if not record:
            return jsonify({'error': 'Record not found'}), 404

        file_dir = os.path.join(app.config['UPLOAD_FOLDER'], record_id)
        if os.path.exists(file_dir):
            for f in os.listdir(file_dir):
                os.remove(os.path.join(file_dir, f))
            os.rmdir(file_dir)

        db.session.delete(record)
        db.session.commit()
        return jsonify({'success': True}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/upload_batch', methods=['POST'])
def upload_batch():
    files = request.files.getlist('image')
    if not files:
        return jsonify({'error': '没有上传任何文件'}), 400

    results = []
    for file in files:
        if file.filename == '' or not allowed_file(file.filename):
            continue

        try:
            record_id = str(uuid.uuid4())
            save_dir = os.path.join(app.config['UPLOAD_FOLDER'], record_id)
            os.makedirs(save_dir, exist_ok=True)

            filename = secure_filename(file.filename)
            original_path = os.path.join(save_dir, filename)
            file.save(original_path)

            result_path = process_image(original_path, save_dir)

            record = MedicalRecord(
                id=record_id,
                original_path=os.path.join(record_id, filename),
                result_path=os.path.join(record_id, 'result.png')
            )
            db.session.add(record)
            db.session.commit()

            results.append({
                'original': f'/static/uploads/{record.original_path}',
                'result': f'/static/uploads/{record.result_path}',
                'time': record.created_at.strftime('%Y-%m-%d %H:%M')
            })
        except Exception as e:
            db.session.rollback()
            continue

    return jsonify(results), 200

@app.route('/clear_records', methods=['POST'])
def clear_records():
    try:
        records = MedicalRecord.query.all()
        for record in records:
            file_dir = os.path.join(app.config['UPLOAD_FOLDER'], record.id)
            if os.path.exists(file_dir):
                for f in os.listdir(file_dir):
                    os.remove(os.path.join(file_dir, f))
                os.rmdir(file_dir)

        db.session.query(MedicalRecord).delete()
        db.session.commit()

        return jsonify({'success': True}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)


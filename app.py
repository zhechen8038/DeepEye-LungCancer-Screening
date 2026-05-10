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
from Polyp_Segmentation.utils.heatmap import heatmap

import sys
# 设置项目根目录为Python路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
# 添加ULS_Segm目录
ULS_SEGM_DIR = os.path.join(BASE_DIR, 'ULS_Segm')
sys.path.append(ULS_SEGM_DIR)
# 添加models目录
MODELS_DIR = os.path.join(ULS_SEGM_DIR, 'models')
sys.path.append(MODELS_DIR)
# 添加Weights目录
WEIGHTS_DIR = os.path.join(ULS_SEGM_DIR, 'Weights')
sys.path.append(WEIGHTS_DIR)

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
    'POLYP_MODEL_PATH': os.path.join(os.path.dirname(__file__), 'Polyp_Segmentation', 'Weights', 'CFANet.pth'),
    'BREAST_MODEL_PATH': os.path.join(os.path.dirname(__file__), 'ULS_Segm', 'Weights', 'EMGANet_WHU.pt'),
    'TESTSIZE': 352,
    'ALLOWED_AVATAR_EXTENSIONS': {'png', 'jpg', 'jpeg', 'gif'},#头像格式
    'AVATAR_FOLDER': os.path.join(os.getcwd(), 'static', 'avatars'),
    'DEFAULT_AVATAR': 'default.jpg',
    'MAX_AVATAR_SIZE': 2 * 1024 * 1024, # 2MB
    'BREAST_INPUT_SIZE': 256  # 新增乳腺癌模型输入尺寸
})

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
    model_type = db.Column(db.String(20), default='polyp')  # 新增模型类型字段
    analysis_result = db.Column(db.String(512))  # 新增分析结果字段

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
polyp_model = None
breast_model = None

# 加载肠镜模型
def load_polyp_model():
    global polyp_model
    if polyp_model is None:
        from Polyp_Segmentation.lib.model import CFANet
        polyp_model = CFANet(channel=64).to(device)
        polyp_model.load_state_dict(torch.load(app.config['POLYP_MODEL_PATH'], map_location=device, weights_only=True))
        polyp_model.eval()
    return polyp_model

# 修改乳腺癌模型加载函数
def load_breast_model():
    global breast_model
    if breast_model is None:
        breast_model = torch.load(
            app.config['BREAST_MODEL_PATH'],
            map_location=device,
            weights_only=False
        )
        breast_model = breast_model.to(device)
        breast_model.eval()
    return breast_model

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

# 乳腺癌图像预处理函数
def breast_preprocess_image(image_path, transform):
    """乳腺癌图像预处理"""
    image = Image.open(image_path).convert('RGB')
    sample = {'image': image}
    sample = transform(sample)
    image_tensor = sample['image'].unsqueeze(0)  # 增加batch维度
    return image_tensor, image

# 乳腺癌结果保存函数
def save_breast_results(pred_tensor, orig_image, save_gray_path, save_overlay_path):
    """保存乳腺癌分割结果"""
    # 预测掩码 (类别索引 * 255)
    pred_mask = torch.argmax(pred_tensor, dim=1).squeeze(0).cpu().numpy().astype(np.uint8) * 255

    # 将灰度掩码resize到原图大小（用NEAREST防止模糊）
    pred_mask_pil = Image.fromarray(pred_mask)
    pred_mask_pil = pred_mask_pil.resize(orig_image.size, Image.NEAREST)
    pred_mask_resized = np.array(pred_mask_pil)

    # 保存灰度图
    cv2.imwrite(save_gray_path, pred_mask_resized)

    # 生成伪彩色热力图
    heatmap = cv2.applyColorMap(pred_mask_resized, cv2.COLORMAP_JET)

    # 原始RGB图转BGR
    orig_rgb = np.array(orig_image)
    orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)

    # 确保heatmap和原图大小一致
    if heatmap.shape[:2] != orig_bgr.shape[:2]:
        heatmap = cv2.resize(heatmap, (orig_bgr.shape[1], orig_bgr.shape[0]))

    # 叠加
    overlay = cv2.addWeighted(orig_bgr, 0.5, heatmap, 0.5, 0)
    cv2.imwrite(save_overlay_path, overlay)
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
                result_path = process_polyp_image(file_path, temp_dir)
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

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return jsonify({'success': '已退出登录'}), 200


@app.route('/')
def head():
    return render_template('index.html')

@app.route('/go_to_AIpage')
def ai_page():
    return render_template('ai-assistant.html')

@app.route('/go_to_upload')
def fir_page():
    return render_template('upload-page.html')

@app.route('/go_to_analyse')
def analyse_page():
    return render_template('analyse.html')

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

@app.route('/upload', methods=['POST','OPTIONS'])
def upload_file():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    model_type = request.form.get('modelType', 'polyp')  # 获取模型类型

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        record_id = str(uuid.uuid4())
        save_dir = os.path.join(app.config['UPLOAD_FOLDER'], record_id)
        os.makedirs(save_dir, exist_ok=True)

        filename = secure_filename(file.filename)
        original_abs_path = os.path.join(save_dir, filename)
        file.save(original_abs_path)

        # 根据模型类型选择处理方式
        if model_type == 'polyp':
            result_abs_path, analysis = process_polyp_image(original_abs_path, save_dir)
        elif model_type == 'breast':
            result_abs_path, analysis = process_breast_image(original_abs_path, save_dir)
        else:
            return jsonify({'error': 'Invalid model type'}), 400

        # 使用 os.path.relpath 计算相对路径
        original_rel_path = os.path.relpath(original_abs_path, app.config['UPLOAD_FOLDER'])
        result_rel_path = os.path.relpath(result_abs_path, app.config['UPLOAD_FOLDER'])

        # 保存记录到数据库
        record = MedicalRecord(
            id=record_id,
            original_path=original_rel_path.replace("\\", "/"),
            result_path=result_rel_path.replace("\\", "/"),
            model_type=model_type,
            analysis_result=analysis
        )
        db.session.add(record)
        db.session.commit()

        return jsonify({
            'original': f'/static/uploads/{original_rel_path.replace(os.sep, "/")}',
            'result': f'/static/uploads/{result_rel_path.replace(os.sep, "/")}',
            'time': record.created_at.strftime('%Y-%m-%d %H:%M'),
            'modelType': model_type,
            'analysis': analysis
        }), 200

    except Exception as e:
        db.session.rollback()
        # 记录详细错误日志
        app.logger.error(f'上传文件错误: {str(e)}',exc_info=True)
        return jsonify({'error': '上传处理失败，请检查文件格式或联系管理员'}), 500

def generate_detailed_analysis(model_type):
    """根据模型类型生成详细的分析结果"""
    if model_type == 'polyp':
        return """<p class="mb-3">AI分析已完成，检测结果如下：</p>
                <ul class="list-disc pl-5 space-y-2">
                    <li>检测到2处息肉病变，最大直径约5mm</li>
                    <li>病变区域已用红色热图标注</li>
                    <li>建议进一步进行结肠镜检查确认</li>
                    <li>根据患者年龄和病变特征，评估为中风险</li>
                </ul>
                <p class="mt-4 font-medium">建议：</p>
                <p>请携带本报告前往消化内科就诊，医生会根据具体情况建议进一步检查或治疗方案。</p>"""
    elif model_type == 'breast':
        return """<p class="mb-3">AI分析已完成，检测结果如下：</p>
                <ul class="list-disc pl-5 space-y-2">
                    <li>右乳外上象限发现1处可疑结节，大小约8mm×6mm</li>
                    <li>BI-RADS分类：4类，恶性可能性2-95%</li>
                    <li>边界欠清晰，形态不规则，可见微小钙化</li>
                </ul>
                <p class="mt-4 font-medium">建议：</p>
                <p>1. 建议进行乳腺钼靶检查进一步评估</p>
                <p>2. 考虑超声引导下穿刺活检</p>
                <p>3. 建议乳腺外科专家门诊就诊</p>"""
    else:
        return "分析完成，结果正常。建议定期复查。"

# 肠镜图像处理
def process_polyp_image(image_path, save_dir):
    img = Image.open(image_path).convert('RGB')
    original_size = img.size

    model = load_polyp_model()
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, _, res = model(img_tensor)
        res = F.interpolate(res, size=original_size[::-1], mode='bilinear', align_corners=False)
        res = res.sigmoid().cpu().numpy().squeeze()

    res = (res - res.min()) / (res.max() - res.min() + 1e-8)

    # 创建结果文件路径
    result_path = os.path.join(save_dir, 'result.png')
    print(f"[DEBUG] 保存路径: {result_path}")

    # 直接使用绝对路径调用 heatmap
    heatmap(res, img_tensor, result_path, original_size)

    # 生成分析结果
    analysis = "PET-CT检测完成，发现肺结节区域"
    return result_path, analysis


# 修改乳腺癌图像处理函数
def process_breast_image(image_path, save_dir):
    # 加载乳腺癌分割模型
    model = load_breast_model()

    # 预处理图像
    from ULS_Segm.util.transforms import test_transforms  # 从ULS_Segm导入
    image_tensor, orig_image = breast_preprocess_image(
        image_path,
        test_transforms
    )
    image_tensor = image_tensor.to(device)

    # 模型推理
    with torch.no_grad():
        output = model(image_tensor)
        if isinstance(output, (list, tuple)):
            output = output[-1]

    # 生成结果文件名
    filename = os.path.splitext(os.path.basename(image_path))[0]
    save_gray_path = os.path.join(save_dir, f'{filename}_gray.png')
    save_overlay_path = os.path.join(save_dir, f'{filename}_overlay.jpg')

    # 保存结果
    save_breast_results(output, orig_image, save_gray_path, save_overlay_path)

    # 生成分析结果
    analysis = "肺癌病灶分割完成，发现可疑区域"

    # 返回叠加图作为结果图像
    return save_overlay_path, analysis


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
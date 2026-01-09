# YouTube 评论情感分析系统

基于 ModernBERT-large 的 YouTube 视频评论情感分析全栈应用。支持中英文及多语言评论分析，自动分类为积极、消极、中性三种情感类型。

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Vue](https://img.shields.io/badge/vue-3.5-green.svg)

## 项目预览

- 输入 YouTube 视频链接
- 自动拉取评论并进行情感分析
- 展示情感分布统计（百分比）
- 显示前10条评论示例及其情感标签

## 主要特性

- **ModernBERT-large 模型** - 高精度多语言情感分析
- **Redis 缓存** - 相同视频1小时内秒级响应
- **多语言支持** - 支持中文、英文等多种语言
- **可视化展示** - 直观的统计数据和评论列表
- **链接验证** - 自动验证 YouTube 链接有效性
- **异步处理** - 进度条实时反馈分析状态
- **API限流保护** - 避免 YouTube API 配额超限

## 🛠技术栈

### 前端
- Vue 3
- Vite
- Axios - HTTP 客户端

### 后端
- FastAPI 
- PyTorch 
- Transformers - Hugging Face 模型库
- Redis
- Google YouTube Data API v3

### 机器学习
- ModernBERT-large - 情感分类模型
- 自训练的三分类模型（消极/中性/积极）

## 系统要求

- Python 3.11+
- Node.js 16+
- Redis 6.0+
- GPU (可选，推荐用于加速推理)

## 快速开始
#### 安装依赖

手动安装：
```bash
pip install fastapi uvicorn google-api-python-client transformers torch emoji redis
```

#### 配置 YouTube API 密钥

1. 访问 [Google Cloud Console](https://console.cloud.google.com/)
2. 创建新项目或选择现有项目
3. 启用 **YouTube Data API v3**
4. 创建凭据 → API 密钥
5. 在 `base.py` 第28行替换：

```python
YOUTUBE_API_KEY = "你的API密钥"
```

#### 放置训练好的模型

将训练好的模型文件放在以下目录：
```
backend/
└── models/
    └── modernBERT-multilingual-finetune/
        ├── config.json
        ├── model.safetensors
        ├── tokenizer.json
        ├── tokenizer_config.json
        └── special_tokens_map.json
```

#### 启动 Redis

**Windows:**
```bash
redis-server
```

**Linux/Mac:**
```bash
redis-server
# 或
sudo service redis-server start
```

验证 Redis 启动：
```bash
redis-cli ping
# 应返回: PONG
```

#### 启动后端服务

```bash
python base.py
```

后端将运行在 `http://localhost:8000`

### 3. 前端设置

#### 安装依赖

```bash
cd frontend
npm install
```

#### 启动开发服务器

```bash
npm run dev
```

前端将运行在 `http://localhost:5173`

### 4. 访问应用

打开浏览器访问 `http://localhost:5173`

## 📖 使用说明

### 分析视频评论

1. 在输入框中粘贴 YouTube 视频链接
    - 支持格式：`https://www.youtube.com/watch?v=xxxxx`
    - 支持格式：`https://youtu.be/xxxxx`

2. 点击 **"分析"** 按钮

3. 等待进度条完成（约5-10秒）

4. 查看分析结果：
    - 三个情感类别的百分比
    - 前10条评论示例及其情感标签

### API 使用

#### 分析视频

```bash
POST http://localhost:8000/api/analyze
Content-Type: application/json

{
  "videoUrl": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
}
```

**响应示例：**
```json
{
  "positive": 45,
  "negative": 25,
  "neutral": 30,
  "total_comments": 200,
  "comment_samples": [
    {
      "text": "Great video!",
      "sentiment": "positive"
    }
  ]
}
```

#### 健康检查

```bash
GET http://localhost:8000/api/health
```

#### 清除缓存

```bash
# 清除指定视频缓存
DELETE http://localhost:8000/api/cache/{video_id}

# 清除所有缓存
DELETE http://localhost:8000/api/cache
```

## 配置说明

### 后端配置 (`base.py`)

```python
# YouTube API
YOUTUBE_API_KEY = "你的API密钥"

# 模型路径
MODEL_PATH = "./models/modernBERT-multilingual-finetune"

# 文本最大长度
MAX_LENGTH = 256

# Redis 缓存过期时间（秒）
CACHE_EXPIRY = 3600  # 1小时

# Redis 连接配置
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
```

### 标签映射

```python
id2label = {
    0: 'negative',  # 消极
    1: 'neutral',   # 中性
    2: 'positive'   # 积极
}
```

## 性能优化

### Redis 缓存策略

- **缓存键格式**: `youtube_sentiment:{video_id}`
- **过期时间**: 1小时（3600秒）
- **优势**:
    - 节省 YouTube API 配额
    - 减少模型推理次数
    - 秒级响应缓存命中

### 批量推理优化

- 批次大小: 32
- 使用 GPU 加速（如果可用）
- 动态 padding 减少计算量

## 开发指南

### 项目结构

```
youtube-sentiment-analysis/
├── frontend/                 # Vue 前端
│   ├── src/
│   │   ├── App.vue          # 主应用组件
│   │   └── main.js
│   ├── package.json
│   └── vite.config.js
│
├── backend/                  # FastAPI 后端
│   ├── base.py              # 主应用文件
│   ├── test_model.py        # 模型测试脚本
│   ├── models/              # 训练好的模型
│   │   └── modernBERT-multilingual-finetune/
│   └── requirements.txt
│
└── README.md
```

### 添加新功能

1. **前端**: 修改 `frontend/src/App.vue`
2. **后端**: 修改 `backend/base.py`
3. **API端点**: 在 `base.py` 中添加新的路由

### 测试

#### 测试后端

```bash
# 测试模型加载
python test_model.py

# 测试 API
curl http://localhost:8000/api/health
```

#### 前端构建

```bash
cd frontend
npm run build
```

## 故障排除

### 问题: 模型加载失败

**解决方案:**
- 检查模型文件是否完整
- 确认路径配置正确
- 查看控制台错误信息

### 问题: Redis 连接失败

**解决方案:**
```bash
# 检查 Redis 是否运行
redis-cli ping

# 重启 Redis
redis-server
```

### 问题: YouTube API 配额超限

**解决方案:**
- 使用 Redis 缓存减少 API 调用
- 等待配额重置（每天午夜太平洋时间）
- 申请更高的配额限制

### 问题: 前端无法连接后端

**解决方案:**
- 检查后端是否启动在 8000 端口
- 检查 CORS 配置
- 查看浏览器控制台错误

## API 配额说明

YouTube Data API v3 免费配额：
- **每日配额**: 10,000 单位
- **commentThreads.list**: 1 单位/请求
- **约可分析**: 100-200个视频/天

建议：
- 启用 Redis 缓存
- 避免重复分析相同视频
- 监控 API 使用情况


## 致谢

- [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-large) - 基础模型
- [FastAPI](https://fastapi.tiangolo.com/) - Web 框架
- [Vue.js](https://vuejs.org/) - 前端框架
- YouTube Data API - 评论数据源

## 更新日志

### v1.0.0 (2026-01-09)
- 初始版本发布
- ModernBERT-large 情感分析
- Redis 缓存支持
- 多语言评论支持
- 前端可视化展示

---

**⭐ 如果这个项目对你有帮助，请给个 Star！**
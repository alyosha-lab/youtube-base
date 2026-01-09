"""
依赖安装: pip install fastapi uvicorn google-api-python-client transformers torch emoji redis
注意: 不要使用 modelscope，使用 transformers 来加载本地模型
需要先启动Redis: redis-server (默认端口6379)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from googleapiclient.discovery import build
from typing import List, Dict
import re
import emoji
import torch
import os
import json
import redis
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from urllib.parse import urlparse, parse_qs

app = FastAPI(title="YouTube情感分析API")

# CORS配置 - 允许前端跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境改为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ 配置部分 ============
YOUTUBE_API_KEY = "YOUR_API_KEY_HERE"  # 请替换为你自己的API密钥（不要暴露在代码中）
MODEL_PATH = os.path.abspath("./models/modernBERT-multilingual-finetune")  # 使用绝对路径
MAX_LENGTH = 256
CACHE_EXPIRY = 3600  # Redis缓存过期时间：1小时（3600秒）

# Redis配置
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# 标签映射（与训练时一致）
id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}

# 初始化Redis连接
try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True,  # 自动解码为字符串
        socket_connect_timeout=5
    )
    # 测试连接
    redis_client.ping()
    print(f"✓ Redis连接成功: {REDIS_HOST}:{REDIS_PORT}")
except redis.ConnectionError:
    print("⚠ 警告: 无法连接到Redis，将不使用缓存功能")
    print("  如需使用缓存，请先启动Redis: redis-server")
    redis_client = None
except Exception as e:
    print(f"⚠ Redis连接警告: {str(e)}")
    redis_client = None

# 检查模型路径
if not os.path.exists(MODEL_PATH):
    print(f"❌ 错误: 模型路径不存在: {MODEL_PATH}")
    print("请确保模型文件在正确的位置")
    raise FileNotFoundError(f"模型路径不存在: {MODEL_PATH}")

# 检查必要文件
required_files = ['config.json', 'model.safetensors']
for file in required_files:
    file_path = os.path.join(MODEL_PATH, file)
    if not os.path.exists(file_path):
        print(f"❌ 错误: 缺少文件 {file}")
        raise FileNotFoundError(f"缺少模型文件: {file}")

print(f"✓ 模型路径验证通过: {MODEL_PATH}")

# 全局模型加载（启动时加载一次）
print(f"正在从 {MODEL_PATH} 加载模型...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"✓ 模型加载完成，使用设备: {device}")
except Exception as e:
    print(f"❌ 模型加载失败: {str(e)}")
    raise


# ============ 数据模型 ============
class VideoRequest(BaseModel):
    videoUrl: str

    @validator('videoUrl')
    def validate_youtube_url(cls, v):
        """验证是否为有效的YouTube链接"""
        if not is_valid_youtube_url(v):
            raise ValueError('无效的YouTube视频链接')
        return v


class CommentSample(BaseModel):
    text: str
    sentiment: str


class AnalysisResponse(BaseModel):
    positive: int
    negative: int
    neutral: int
    total_comments: int
    comment_samples: List[CommentSample]


# ============ Redis缓存函数 ============
def get_cache_key(video_id: str) -> str:
    """生成Redis缓存键"""
    return f"youtube_sentiment:{video_id}"


def get_cached_result(video_id: str) -> Dict:
    """从Redis获取缓存的分析结果"""
    if not redis_client:
        return None

    try:
        cache_key = get_cache_key(video_id)
        cached_data = redis_client.get(cache_key)

        if cached_data:
            print(f"✓ 缓存命中: {video_id}")
            return json.loads(cached_data)

        print(f"○ 缓存未命中: {video_id}")
        return None
    except Exception as e:
        print(f"⚠ Redis读取错误: {str(e)}")
        return None


def set_cached_result(video_id: str, result: Dict):
    """将分析结果存入Redis缓存"""
    if not redis_client:
        return

    try:
        cache_key = get_cache_key(video_id)
        redis_client.setex(
            cache_key,
            CACHE_EXPIRY,
            json.dumps(result, ensure_ascii=False)
        )
        print(f"✓ 结果已缓存: {video_id} (过期时间: {CACHE_EXPIRY}秒)")
    except Exception as e:
        print(f"⚠ Redis写入错误: {str(e)}")


def clear_cache(video_id: str = None):
    """清除缓存（可选指定视频ID，否则清除所有）"""
    if not redis_client:
        return False

    try:
        if video_id:
            cache_key = get_cache_key(video_id)
            redis_client.delete(cache_key)
            print(f"✓ 缓存已清除: {video_id}")
        else:
            # 清除所有相关缓存
            pattern = "youtube_sentiment:*"
            keys = redis_client.keys(pattern)
            if keys:
                redis_client.delete(*keys)
                print(f"✓ 已清除 {len(keys)} 个缓存")
        return True
    except Exception as e:
        print(f"⚠ 清除缓存错误: {str(e)}")
        return False


# ============ 工具函数 ============
def is_valid_youtube_url(url: str) -> bool:
    """判断是否为有效的YouTube链接"""
    youtube_patterns = [
        r'(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]+',
        r'(https?://)?(www\.)?youtube\.com/watch\?.*v=[\w-]+'
    ]
    return any(re.match(pattern, url) for pattern in youtube_patterns)


def extract_video_id(url: str) -> str:
    """从YouTube链接中提取视频ID"""
    parsed_url = urlparse(url)

    # 处理 youtube.com/watch?v=xxx 格式
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        query_params = parse_qs(parsed_url.query)
        return query_params.get('v', [None])[0]

    # 处理 youtu.be/xxx 格式
    elif parsed_url.hostname == 'youtu.be':
        return parsed_url.path.lstrip('/')

    return None


def fetch_youtube_comments(video_id: str, max_results: int = 200) -> List[str]:
    """
    使用YouTube Data API v3获取评论
    返回评论文本列表
    """
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

        comments = []
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=min(max_results, 100),  # API单次最多100条
            textFormat='plainText'
        )

        while request and len(comments) < max_results:
            response = request.execute()

            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)

                if len(comments) >= max_results:
                    break

            # 获取下一页
            request = youtube.commentThreads().list_next(request, response)

        return comments[:max_results]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取评论失败: {str(e)}")


def clean_text(text: str) -> str:
    """
    清洗文本数据
    - 移除emoji表情
    - 移除多余空格
    - 移除URL
    """
    # 移除emoji
    text = emoji.replace_emoji(text, replace='')

    # 移除URL
    text = re.sub(r'http\S+|www\S+', '', text)

    # 移除多余空格和换行
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def classify_comments(comments: List[str]) -> Dict:
    """
    批量分类评论
    返回: {
        'positive': x%,
        'negative': y%,
        'neutral': z%,
        'total_comments': n,
        'comment_samples': [前10条评论及其情感]
    }
    """
    if not comments:
        return {
            'positive': 0,
            'negative': 0,
            'neutral': 0,
            'total_comments': 0,
            'comment_samples': []
        }

    # 清洗文本
    cleaned_comments = [clean_text(comment) for comment in comments]

    # 保存原始评论和清洗后的评论的映射
    comment_pairs = [(orig, cleaned) for orig, cleaned in zip(comments, cleaned_comments) if cleaned.strip()]

    if not comment_pairs:
        return {
            'positive': 0,
            'negative': 0,
            'neutral': 0,
            'total_comments': 0,
            'comment_samples': []
        }

    # 只对清洗后的非空评论进行预测
    valid_cleaned = [cleaned for _, cleaned in comment_pairs]

    # 批量预测（提高效率）
    batch_size = 32
    predictions = []

    for i in range(0, len(valid_cleaned), batch_size):
        batch = valid_cleaned[i:i+batch_size]

        # Tokenize
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 预测
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            batch_predictions = torch.argmax(logits, dim=1).cpu().tolist()
            predictions.extend(batch_predictions)

    # 统计结果 (0: negative, 1: neutral, 2: positive)
    sentiment_counts = {
        'negative': predictions.count(0),
        'neutral': predictions.count(1),
        'positive': predictions.count(2)
    }

    total = len(predictions)

    # 生成前10条评论示例（使用原始评论文本）
    comment_samples = []
    for i in range(min(10, len(comment_pairs))):
        original_comment, _ = comment_pairs[i]
        sentiment = id2label[predictions[i]]
        comment_samples.append({
            'text': original_comment,
            'sentiment': sentiment
        })

    # 转换为百分比
    return {
        'positive': round(sentiment_counts['positive'] / total * 100),
        'negative': round(sentiment_counts['negative'] / total * 100),
        'neutral': round(sentiment_counts['neutral'] / total * 100),
        'total_comments': total,
        'comment_samples': comment_samples
    }


# ============ API路由 ============
@app.get("/")
def root():
    return {"message": "YouTube情感分析API正在运行", "status": "ok"}


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_video(request: VideoRequest):
    """
    分析YouTube视频评论的情感倾向（带Redis缓存）
    """
    try:
        # 1. 提取视频ID
        video_id = extract_video_id(request.videoUrl)
        if not video_id:
            raise HTTPException(status_code=400, detail="无法解析视频ID")

        # 2. 检查Redis缓存
        cached_result = get_cached_result(video_id)
        if cached_result:
            print(f"→ 返回缓存结果: {video_id}")
            return cached_result

        # 3. 获取评论
        print(f"正在获取视频 {video_id} 的评论...")
        comments = fetch_youtube_comments(video_id, max_results=200)

        if not comments:
            raise HTTPException(status_code=404, detail="该视频没有评论或评论已关闭")

        print(f"成功获取 {len(comments)} 条评论")

        # 4. 情感分析
        print("开始情感分析...")
        results = classify_comments(comments)

        print(f"分析完成: 积极{results['positive']}% 消极{results['negative']}% 中性{results['neutral']}%")

        # 5. 存入Redis缓存
        set_cached_result(video_id, results)

        return results

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析过程出错: {str(e)}")


@app.get("/api/health")
def health_check():
    """健康检查接口"""
    redis_status = "connected" if redis_client else "disconnected"

    # 获取Redis信息
    redis_info = {}
    if redis_client:
        try:
            redis_info = {
                "status": "connected",
                "cache_keys": len(redis_client.keys("youtube_sentiment:*")),
                "memory_used": redis_client.info("memory").get("used_memory_human", "N/A")
            }
        except:
            redis_info = {"status": "error"}

    return {
        "status": "healthy",
        "model_path": MODEL_PATH,
        "device": str(device),
        "labels": id2label,
        "redis": redis_info,
        "cache_expiry": f"{CACHE_EXPIRY}秒 (1小时)"
    }


@app.delete("/api/cache/{video_id}")
async def delete_cache(video_id: str):
    """
    删除指定视频的缓存
    """
    if clear_cache(video_id):
        return {"message": f"缓存已清除: {video_id}"}
    else:
        raise HTTPException(status_code=500, detail="清除缓存失败")


@app.delete("/api/cache")
async def clear_all_cache():
    """
    清除所有缓存
    """
    if clear_cache():
        return {"message": "所有缓存已清除"}
    else:
        raise HTTPException(status_code=500, detail="清除缓存失败")


@app.post("/api/test-predict")
async def test_predict(text: str):
    """
    测试单个文本预测（用于调试）
    """
    try:
        cleaned_text = clean_text(text)

        inputs = tokenizer(
            cleaned_text,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()

        prediction = id2label[pred_id]

        return {
            "original_text": text,
            "cleaned_text": cleaned_text,
            "prediction": prediction
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import os
import uuid
import time
from threading import Thread, Lock
from collections import deque

app = Flask(__name__)
CORS(app)

# 尝试导入 LaMa 封装
try:
    from lama_runner import inpaint as lama_inpaint, LamaNotReady  # type: ignore
except Exception:  # pragma: no cover
    lama_inpaint = None
    class LamaNotReady(Exception): ...


# ===== 队列系统 =====
task_queue = deque()  # 任务队列
task_results = {}  # 任务结果存储 {task_id: {"status": "processing|done|error", "result": data}}
queue_lock = Lock()
processing = False

def process_queue():
    """后台线程处理队列"""
    global processing
    while True:
        with queue_lock:
            if len(task_queue) == 0:
                processing = False
                time.sleep(0.5)
                continue
            
            processing = True
            task = task_queue.popleft()
        
        task_id = task['id']
        try:
            # 更新状态为处理中
            task_results[task_id]['status'] = 'processing'
            task_results[task_id]['queue_position'] = 0
            
            # 执行实际处理
            image = task['image']
            mask = task['mask']
            engine = task['engine']
            
            # LaMa 处理逻辑
            if (engine in ['lama', 'auto']) and lama_inpaint is not None:
                try:
                    out_img = lama_inpaint(image, mask)
                except LamaNotReady:
                    out_img = None
            else:
                out_img = None
            
            if out_img is None:
                # OpenCV Telea 兜底
                img_np = np.array(image)
                mask_np = np.array(mask)
                _, mask_bin = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
                result = cv2.inpaint(img_np, (mask_bin > 0).astype(np.uint8), 3, cv2.INPAINT_TELEA)
                out_img = Image.fromarray(result)
            
            # 保存结果
            buf = BytesIO()
            out_img.save(buf, format='PNG')
            buf.seek(0)
            
            task_results[task_id]['status'] = 'done'
            task_results[task_id]['result'] = buf.getvalue()
            
        except Exception as e:
            task_results[task_id]['status'] = 'error'
            task_results[task_id]['error'] = str(e)
        
        # 更新队列中其他任务的位置
        with queue_lock:
            for i, t in enumerate(task_queue):
                task_results[t['id']]['queue_position'] = i + 1

# 启动后台处理线程
Thread(target=process_queue, daemon=True).start()


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


@app.route('/api/inpaint', methods=['POST'])
def inpaint():
    """提交任务到队列"""
    if 'image' not in request.files or 'mask' not in request.files:
        return jsonify({"error": "image and mask are required"}), 400

    image_file = request.files['image']
    mask_file = request.files['mask']
    engine = request.args.get('engine') or os.getenv('INPAINT_ENGINE', 'auto')

    try:
        image = Image.open(image_file.stream).convert('RGB')
        mask = Image.open(mask_file.stream).convert('L')
        
        # 生成任务 ID
        task_id = str(uuid.uuid4())
        
        # 添加到队列
        with queue_lock:
            task_queue.append({
                'id': task_id,
                'image': image,
                'mask': mask,
                'engine': engine,
                'timestamp': time.time()
            })
            queue_position = len(task_queue)
        
        # 初始化任务结果
        task_results[task_id] = {
            'status': 'queued',
            'queue_position': queue_position
        }
        
        return jsonify({
            'task_id': task_id,
            'queue_position': queue_position,
            'message': f'已加入队列，前面还有 {queue_position - 1} 个任务'
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/task/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """查询任务状态"""
    if task_id not in task_results:
        return jsonify({"error": "Task not found"}), 404
    
    task = task_results[task_id]
    
    if task['status'] == 'done':
        # 返回处理后的图片
        return send_file(
            BytesIO(task['result']),
            mimetype='image/png'
        )
    elif task['status'] == 'error':
        return jsonify({"error": task['error']}), 500
    else:
        # 返回状态信息
        return jsonify({
            'status': task['status'],
            'queue_position': task.get('queue_position', 0),
            'message': f'处理中...' if task['status'] == 'processing' else f'排队中，前面还有 {task.get("queue_position", 0)} 个任务'
        })


if __name__ == '__main__':
    # 支持 Railway 和本地开发
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import os

app = Flask(__name__)
CORS(app)

# 尝试导入 LaMa 封装
try:
    from lama_runner import inpaint as lama_inpaint, LamaNotReady  # type: ignore
except Exception:  # pragma: no cover
    lama_inpaint = None
    class LamaNotReady(Exception): ...


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


@app.route('/api/inpaint', methods=['POST'])
def inpaint():
    if 'image' not in request.files or 'mask' not in request.files:
        return jsonify({"error": "image and mask are required"}), 400

    image_file = request.files['image']
    mask_file = request.files['mask']

    engine = request.args.get('engine') or os.getenv('INPAINT_ENGINE', 'auto')

    try:
        image = Image.open(image_file.stream).convert('RGB')
        mask = Image.open(mask_file.stream).convert('L')

        # 优先使用 LaMa（当 engine=lama，或 auto 且 LaMa 可用）
        if (engine in ['lama', 'auto']) and lama_inpaint is not None:
            try:
                out_img = lama_inpaint(image, mask)
            except LamaNotReady as e:
                if engine == 'lama':
                    return jsonify({"error": str(e)}), 500
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

        buf = BytesIO()
        out_img.save(buf, format='PNG')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # 支持 Railway 和本地开发
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

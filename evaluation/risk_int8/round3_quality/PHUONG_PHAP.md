# Vòng 3 — bảo toàn đầu ra FP32

Vòng 2 đã chọn theo macro-F1 DEV nhưng không tổng quát sang bộ 200. Đây là vòng phát triển tiếp theo sau khi đã xem kết quả trước, không phải independent test mới.

## Ba cấu hình, chốt trước đánh giá

1. **W8A32:** embedding từng hàng INT8; trọng số tuyến tính per-output-channel INT8; activation giữ FP32. ONNX DequantizeLinear giải mã weight khi tính toán. Không gọi đây là W8A8. Runtime có thể giải mã/trải lại float weight lúc tối ưu, nên giảm file không đồng nghĩa giảm RAM hoặc tăng tốc tương ứng.
2. **SmoothQuant-inspired alpha 0.5:** lấy max activation theo kênh từ 64 mẫu TRAIN, scale s = max(|X|)^alpha / max(|W|)^(1-alpha). Với MatMul Y=XW, thay bằng (X/s)(sW) trước khi lượng tử hóa. Giữ phép Div tường minh, không giả vờ đã fold và tối ưu đầy đủ như bản triển khai bài báo.
3. **Như trên, alpha 0.7.**

Hai cấu hình smoothing vẫn dùng embedding theo từng hàng và INT8 per-channel cho linear. FP32 trước/sau smoothing được kiểm tra tương đương trên bốn mẫu TRAIN trước khi nén. ID của 64 mẫu calibration có seed cố định, lưu TRAIN_CALIBRATION_IDS.json.

## Chọn cấu hình

Chọn theo độ lệch logits so với teacher FP32 trên 398 DEV, không dùng gold để xếp hạng. Ngân sách file 450 MB, giới hạn thời gian CPU <=1.25 lần FP32; nếu không cấu hình nào đạt thời gian thì chọn phương án nén có MSE nhỏ nhất và báo rõ đánh đổi. Chốt tên/hash vào SELECTED_BEFORE_TEST.json trước khi chạy lại 200 mẫu. Không chỉnh threshold hoặc retrain.

## Nguồn

- SmoothQuant: https://github.com/mit-han-lab/smoothquant
- ONNX per-channel và loại lượng tử hóa: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- Variable Layerwise Quantization (ACL 2025): https://aclanthology.org/2025.findings-acl.29/
- OS+ (EMNLP 2023): https://aclanthology.org/2023.emnlp-main.102/
- BWTA (arXiv 2026, under review; chỉ khảo sát, không triển khai): https://arxiv.org/abs/2604.03957

# Thử nghiệm nén risk-only INT8

Đây là biến thể **nén checkpoint risk public**, vẫn giữ encoder đã học action/semantic. Khác baseline XLM-R chỉ học risk: bỏ chạy model semantic riêng không xóa kiến thức semantic trong trọng số risk.

[Kết quả và các vòng thử nghiệm](../../evaluation/risk_int8/README.md). Bản được chọn: `smooth_alpha_0_5` (SmoothQuant-inspired, không tuyên bố tái hiện đầy đủ SmoothQuant). Phân phối ở Release thử nghiệm riêng; không thay thế checkpoint `model-v1`, API hay app Android hiện có.

## Chạy model nén

```powershell
python -m pip install -r experiments/risk_int8/requirements-int8.txt
python scripts/download_research_assets.py int8
python experiments/risk_int8/risk_only_runtime.py --text "Đừng gửi OTP cho người khác."
```

Runtime chỉ dùng ONNX Runtime, tokenizers và NumPy, không nạp PyTorch/semantic checkpoint. Model ở `artifacts/onnx/risk_int8.onnx`, tokenizer ở `experiments/risk_int8/tokenizer/tokenizer.json`. Model có đầu ra NO_EVIDENCE/REVIEW/HIGH_EVIDENCE. Mặc định 4 CPU threads, batch 1.

## Tái lập quy trình nén

Yêu cầu PyTorch và dependency train trong README chính, checkpoint risk public từ `scripts/download_checkpoints.ps1`, tokenizer tải bằng lệnh ở trên. Các bước sau tạo model lớn và chủ động đánh giá lại bộ 200:

```powershell
python experiments/risk_int8/run_experiment.py
python experiments/risk_int8/audit_preprocessing.py
python experiments/risk_int8/improve_quantization.py
python experiments/risk_int8/quality_preserving_quantization.py
python experiments/risk_int8/benchmark_final.py
```

Vòng đầu export FP32 và dynamic INT8; vòng 2 thử per-channel/embedding theo hàng; vòng 3 thêm weight-only và cân bằng activation/weight. Cân bằng dùng 64 mẫu TRAIN, lựa chọn theo độ gần logits FP32 trên DEV trong vòng 3, không dùng test làm calibration. Test đã được xem từ các vòng trước nên toàn bộ là phát triển thích nghi, không phải independent test mới. Đầu ra mới trong `outputs/`, `round2_research/`, `round3_quality/`; không ghi vào thư mục bằng chứng `evaluation/`.

Các file `source/` là bản chụp kiến trúc/collator dùng trong export gốc. Code được chuyển sang đường dẫn tương đối repo; manifest đã lưu ghi mã băm nguồn ở thời điểm chạy gốc, không phải mã băm script đóng gói sau khi sửa đường dẫn. Không commit deps/cache, model ONNX, logits trung gian hay activation cache.

# Nén risk public sang ONNX/INT8

Thử nghiệm CPU laptop ngày 03/11/2026, **giữ encoder risk đã học action/semantic**, không train lại. Các vòng phát triển và prediction được giữ riêng, gồm cả vòng không cải thiện.

| Phiên bản | Dung lượng MB (thập phân) | Accuracy | Macro-F1 hai nhãn gold | HIGH_EVIDENCE → xanh /70 | NO_EVIDENCE → đỏ /130 |
|---|---:|---:|---:|---:|---:|
| ONNX FP32 | 1111,6 | 76,00% | 81,35% | 9 | 11 |
| INT8 vòng đầu | 278,6 | 75,00% | 77,22% | 13 | 21 |
| Vòng 2: embedding theo hàng + per-channel | 281,2 | 71,00% | 75,51% | 8 | 23 |
| Vòng 3: SmoothQuant-inspired alpha 0,5 | **281,6** | **75,50%** | **81,61%** | **5** | **11** |

Không lựa chọn lại ngưỡng. Vòng 3 giảm dung lượng khoảng 74,66% so FP32; median CPU **37,06 ms → 16,46 ms (~2,25×)**. Hai thứ tự chạy đảo nhau, 400 timed calls/model, 5 warmup/session, 4 threads, batch 1; không gồm tokenizer/guidance, không phải tốc độ Android. Nguồn [FINAL_LATENCY.json](round3_quality/FINAL_LATENCY.json).

## Vì sao FP32 là 76,0%, không phải 76,5% như public gốc?

Các so sánh nén dùng parser hiện tại của API; prediction public gốc dùng parser lịch sử. Có 7/200 nhãn khác nhau, tổng accuracy chênh một mẫu. Audit xác nhận dùng parser lịch sử với ONNX FP32 khớp 200/200 prediction gốc. Không sửa parser để chọn điểm tốt hơn. Vì thế **không ghép bảng này với bảng baseline 57,0%/76,5% như cùng preprocessing**. Xem [PREPROCESSING_AUDIT.json](round1/PREPROCESSING_AUDIT.json).

## Phương pháp

- Vòng 1: dynamic INT8 với ONNX Runtime.
- Vòng 2: word embedding INT8 theo hàng/token; Linear per-output-channel; có thử giữ embedding hoặc hai block cuối FP32.
- Vòng 3: weight-only và SmoothQuant-inspired alpha 0,5/0,7. Scale activation/weight bằng 64 mẫu TRAIN, kiểm tra tương đương FP32 trước nén; chọn trên DEV logit MSE và giới hạn dung lượng/tốc độ. Giữ Div tường minh, không dùng đầy đủ kernel/folding của bài báo SmoothQuant.

## File và kiểm tra

- [Code export/nén/runtime](../../experiments/risk_int8).
- `round1/`: manifest, metric, prediction FP32/INT8 và audit parser.
- `round2_research/`: report, chốt trước test và prediction của vòng 2.
- `round3_quality/`: report, DEV selection, calibration IDs TRAIN, prediction bản được chọn và tốc độ đo lại.
- [Model và tokenizer — Release thử nghiệm](https://github.com/TuanAnhPhan-pika/SCAMGUARD-VN/releases/tag/risk-int8-v1), tải bằng `python scripts/download_research_assets.py int8`; không thay checkpoint chính.
- Kiểm tra lại metric từ prediction đã lưu: `python scripts/verify_research_results.py`.

## Giới hạn

Comparative re-evaluation trên bộ 200 đã xem; các vòng sau có bối cảnh thất bại ở vòng trước. Không phải independent test mới, chưa chứng minh tốt hơn ngoài thực tế. Gold không có REVIEW. INT8 vòng đầu làm tăng lỗi; bản cải thiện giảm một số lỗi nhưng strict accuracy vẫn thấp hơn FP32. Cần dữ liệu mới và kiểm chứng trên thiết bị đích. Báo cáo ở đây chỉ cho laptop; không đưa số đo emulator/điện thoại vào bảng này. API/app public và bản model-v1 không thay đổi.

Tham khảo triển khai: [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html), [SmoothQuant](https://github.com/mit-han-lab/smoothquant).

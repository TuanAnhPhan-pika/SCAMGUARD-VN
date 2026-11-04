# Đối chứng XLM-R chỉ học risk với pipeline public

Train 04/11/2026; bổ sung cách chấm giống public ngày 05/11/2026. **Không thay model, ngưỡng hoặc prediction public**. Hai bản dùng cùng 3.805 risk TRAIN / 398 DEV, cấu trúc classifier và cấu hình giai đoạn risk. Baseline chỉ khởi tạo XLM-R gốc; public đã qua học action/semantic.

| Chỉ số theo cùng cách chấm public | XLM-R chỉ học risk | Public |
|---|---:|---:|
| Strict accuracy | 57,00% (114/200) | 76,50% (153/200) |
| Macro-F1 hai nhãn có gold | 64,57% | 81,79% |
| Operational accuracy | 68,50% | 86,50% |
| Positive được cảnh báo vàng/đỏ | 91,43% (64/70) | 90,00% (63/70) |
| NO_EVIDENCE đoán đúng xanh | 46,15% (60/130) | 73,08% (95/130) |
| NO_EVIDENCE bị cảnh báo vàng/đỏ | 53,85% (70/130) | 26,92% (35/130) |

[Toàn bộ bảng màu, từng nhóm và precision/recall/F1](SO_SANH_DAY_DU.md). [Code train baseline](../../experiments/xlmr_risk_only). [Lịch sử train](../../reports/training/xlmr_risk_only_training_report.json). [Checkpoint baseline — Release thử nghiệm](https://github.com/TuanAnhPhan-pika/SCAMGUARD-VN/releases/tag/baseline-risk-only-v1).

## Bằng chứng và cách chấm

- Input/gold: tham chiếu [bộ 200 gốc](../independent_test_200), không sao chép hoặc sửa dữ liệu.
- `baseline_predictions_frozen.jsonl`: 200 prediction baseline; `PREDICTION_FREEZE.json`: checkpoint/input/prediction SHA256.
- `EXACT_PUBLIC_SCORING.json`: metric đầy đủ của cả hai, dùng nguyên khối code chấm public và biểu thức macro-F1 từ `scripts/verify_release.py`.
- `PRETRAIN_MANIFEST.json`: bằng chứng lịch sử train local; tên đường dẫn có tiền tố `github_publish` phản ánh checkout gốc, không phải đường dẫn bắt buộc để chạy bản đóng gói.
- Strict không gộp REVIEW. Macro-F1 hai nhãn chỉ bỏ REVIEW khỏi phép lấy trung bình vì không có gold REVIEW.
- Operational: POSITIVE chấp nhận REVIEW/HIGH_EVIDENCE; HARD_NEGATIVE chấp nhận NO_EVIDENCE/REVIEW; EASY_NEGATIVE chỉ chấp nhận NO_EVIDENCE.

## Giới hạn

Một seed; test đã được xem, nên đây là comparative re-evaluation, không phải test blind mới. Không chọn lại checkpoint hay ngưỡng theo test. Baseline khởi tạo nguyên bản XLM-R; public kế thừa cả action và semantic pretraining. Chênh lệch không được quy hoàn toàn cho riêng semantic. Recall cảnh báo của baseline cao hơn một mẫu nhưng cảnh báo trên mẫu hợp lệ nhiều hơn. Không có gold REVIEW để đo đầy đủ ba lớp.

Kiểm tra số liệu từ prediction, không suy luận/train lại: `python scripts/verify_research_results.py`.

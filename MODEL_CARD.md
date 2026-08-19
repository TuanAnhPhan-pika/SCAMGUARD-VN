# Model card

## Mô tả

ScamGuard-VN gồm hai checkpoint XLM-R:

1. Semantic multi-task model dự đoán 10 `CandidateAction` và bốn thuộc tính `Requested`, `Negated`, `Quoted`, `Reported`.
2. End-to-end risk model dự đoán `NO_EVIDENCE`, `REVIEW`, `HIGH_EVIDENCE`, khởi tạo encoder từ semantic checkpoint rồi fine-tune phần trên.

## Intended use

- Nghiên cứu nhận diện lừa đảo tiếng Việt.
- Hỗ trợ cảnh báo trong ứng dụng thử nghiệm.
- Phân tích lỗi semantic/pragmatic và hard negative.

Không dùng model như bằng chứng duy nhất để khóa tài khoản, cáo buộc cá nhân, từ chối dịch vụ hoặc đưa ra quyết định tài chính/pháp lý.

## Dữ liệu và metric

Xem `data/DATASET_CARD.md`, `reports/training` và `evaluation/independent_test_200`. DEV metric cao hơn đáng kể so với independent test; người dùng cần báo cáo cả hai và không diễn giải DEV như hiệu năng ngoài thực tế.

## Hạn chế đã biết

- Nhạy với domain shift, OCR lỗi và social context ngầm.
- Hard negative vẫn tạo false-red.
- CandidateAction là taxonomy đóng.
- Confidence là development score, chưa phải xác suất scam được calibration ngoài thực tế.
- Guidance do mô hình sinh là tầng tùy chọn, không phải đầu ra kiểm chứng của classifier.

## Checkpoint integrity

| File | SHA256 |
|---|---|
| `checkpoint_best.pt` | `ae0420e7efbdbe791b71907131c480e6fcb9d7cef9f1e4f6382cdc392eb66019` |
| `risk_e2e_best.pt` | `824b58d32366d746adc32b4350c73cad49054f538d365f6661da4036535c8328` |


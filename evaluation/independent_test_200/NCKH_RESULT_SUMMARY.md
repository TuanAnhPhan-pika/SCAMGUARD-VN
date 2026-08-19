# Kết quả kiểm thử độc lập ScamGuard-VN

## Điều kiện kiểm thử

- Số trường hợp: 200, gồm 70 positive, 65 hard negative và 65 easy negative.
- Cả 200 trường hợp đều không xuất hiện nguyên văn trong TRAIN hoặc DEV.
- Semantic checkpoint, risk checkpoint, preprocessing và quy tắc argmax được đóng băng trước suy luận.
- Mô hình chỉ đọc `FINAL_MODEL_INPUT.jsonl`.
- Prediction được lưu và băm SHA256 trước khi mở `INTERNAL_GOLD.jsonl`.
- Không huấn luyện, hiệu chỉnh ngưỡng, thay đổi nhãn hoặc chạy lại sau khi xem gold.

## Kết quả

| Chỉ số | Kết quả |
|---|---:|
| Strict accuracy | 76,50% |
| Strict macro-F1 trên 2 nhãn xuất hiện trong gold | 81,79% |
| Strict weighted-F1 | 81,82% |
| Operational accuracy | 86,50% |
| Positive được cảnh báo đỏ hoặc vàng | 63/70 (90,00%) |
| Hard negative không bị báo đỏ | 52/65 (80,00%) |
| Easy negative được xếp xanh | 58/65 (89,23%) |

Operational accuracy áp dụng chính sách của hệ thống hỗ trợ: positive chấp nhận `REVIEW` hoặc `HIGH_EVIDENCE`; hard negative chấp nhận `NO_EVIDENCE` hoặc `REVIEW`; easy negative yêu cầu `NO_EVIDENCE`.

Gold không chứa lớp `REVIEW`, vì vậy macro-F1 ba lớp 54,53% được lưu để minh bạch nhưng không nên dùng làm chỉ số đại diện chính. Chỉ số này phạt mô hình trên một lớp có support bằng 0.

## Trạng thái hợp lệ

`FINAL_INDEPENDENT_TEST_VALID = YES`

- `MODEL_MODIFIED_AFTER_TEST_START = NO`
- `THRESHOLD_TUNED_AFTER_TEST_START = NO`
- `GOLD_OPENED_BEFORE_PREDICTION_FREEZE = NO`
- `TEST_CASES_CHANGED_AFTER_TEST_START = NO`
- `PREDICTIONS_FROZEN_BEFORE_GOLD = YES`

Sau lần đánh giá này, bộ 200 trường hợp đã trở thành benchmark khóa. Nếu mô hình được sửa hoặc huấn luyện lại, mọi lần chạy tiếp theo trên bộ này phải được mô tả là comparative re-evaluation, không phải independent test mới.

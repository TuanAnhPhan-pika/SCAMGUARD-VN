# Dataset card

## Phạm vi

Dữ liệu phục vụ nhận diện hành động và rủi ro trong tin nhắn/hội thoại tiếng Việt. Bốn file công bố là:

| File | Số mẫu | Mục đích |
|---|---:|---|
| `SEMANTIC_TRAIN.jsonl` | 4.717 | Action và bốn thuộc tính ngữ dụng |
| `SEMANTIC_DEV.jsonl` | 486 | Chọn checkpoint semantic |
| `RISK_TRAIN.jsonl` | 3.805 | Ba mức rủi ro |
| `RISK_DEV.jsonl` | 398 | Chọn checkpoint risk |

## Nhãn

`candidate_action`: `NONE`, `TRANSFER_VALUE`, `DISCLOSE_SECRET`, `INSTALL_SOFTWARE`, `GRANT_DEVICE_ACCESS`, `MOVE_CONVERSATION`, `FOLLOW_LINK`, `SEND_MESSAGE`, `PARTICIPATE_TASK`, `SELECT_OPTION`.

Thuộc tính ngữ dụng: `requested`, `negated`, `quoted`, `reported`, mỗi thuộc tính nhận `YES/NO`.

Nhãn risk: `NO_EVIDENCE`, `REVIEW`, `HIGH_EVIDENCE`.

## Nguồn và kiểm soát

Dữ liệu tổng hợp từ nhiều đợt xây dựng: mẫu do người viết, hội thoại tổng hợp có kiểm soát, hard negative và counterfactual family. Các mẫu trong cùng scenario/pair được giữ cùng split để giảm leakage. Đợt bổ sung high-value có 203/204 mẫu qua audit tự động và được đánh dấu chỉ dùng TRAIN; chúng không được xem là human gold.

## Hạn chế

- Dữ liệu tổng hợp còn nhiều và có thể mang phong cách của mô hình sinh.
- `NONE` là lớp đóng nên chưa bao phủ mọi hành động đời sống.
- Independent gold hiện không chứa `REVIEW`.
- OCR noise, phương ngữ, viết tắt mới và social context lạ vẫn là domain gap.
- Cần human review độc lập trước khi dùng cho quyết định ảnh hưởng người thật.

Không dùng dữ liệu để tự động cáo buộc cá nhân/tổ chức là lừa đảo.


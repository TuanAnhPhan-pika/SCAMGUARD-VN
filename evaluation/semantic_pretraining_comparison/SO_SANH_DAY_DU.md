# So sánh đầy đủ XLM-R chỉ học risk và model public

Cùng 200 case: 130 gold NO_EVIDENCE (65 easy negative, 65 hard negative), 70 gold HIGH_EVIDENCE. Không có gold REVIEW. Xanh = NO_EVIDENCE; vàng = REVIEW; đỏ = HIGH_EVIDENCE. Tỷ lệ trong mỗi hàng được chia cho số mẫu gold của nhóm đó, không chia cho số dự đoán.

## Đầu ra theo nhãn gold

| Nhóm gold | Đầu ra | XLM-R chỉ học risk | Public |
|---|---|---:|---:|
| NO_EVIDENCE (130) | Xanh | 60/130 (46.15%) | 95/130 (73.08%) |
| NO_EVIDENCE (130) | Vàng | 35/130 (26.92%) | 21/130 (16.15%) |
| NO_EVIDENCE (130) | Đỏ | 35/130 (26.92%) | 14/130 (10.77%) |
| NO_EVIDENCE (130) | Vàng hoặc đỏ | 70/130 (53.85%) | 35/130 (26.92%) |
| HIGH_EVIDENCE (70) | Xanh | 6/70 (8.57%) | 7/70 (10.00%) |
| HIGH_EVIDENCE (70) | Vàng | 10/70 (14.29%) | 5/70 (7.14%) |
| HIGH_EVIDENCE (70) | Đỏ | 54/70 (77.14%) | 58/70 (82.86%) |
| HIGH_EVIDENCE (70) | Vàng hoặc đỏ | 64/70 (91.43%) | 63/70 (90.00%) |

Public giảm cảnh báo vàng/đỏ trên gold NO_EVIDENCE từ 70 xuống 35 mẫu: giảm 35 mẫu, tương ứng 26,92 điểm phần trăm. Riêng cảnh báo đỏ giảm từ 35 xuống 14 mẫu. Trên HIGH_EVIDENCE, baseline cảnh báo vàng/đỏ 64/70 so với public 63/70; public có thêm một mẫu bị đoán xanh, nhưng nhận đúng đỏ nhiều hơn bốn mẫu.

## Chi tiết ba nhóm đánh giá

| Nhóm | Đầu ra | XLM-R chỉ học risk | Public |
|---|---|---:|---:|
| EASY_NEGATIVE (65) | Xanh | 39/65 (60.00%) | 58/65 (89.23%) |
| EASY_NEGATIVE (65) | Vàng | 22/65 (33.85%) | 6/65 (9.23%) |
| EASY_NEGATIVE (65) | Đỏ | 4/65 (6.15%) | 1/65 (1.54%) |
| EASY_NEGATIVE (65) | Vàng hoặc đỏ | 26/65 (40.00%) | 7/65 (10.77%) |
| HARD_NEGATIVE (65) | Xanh | 21/65 (32.31%) | 37/65 (56.92%) |
| HARD_NEGATIVE (65) | Vàng | 13/65 (20.00%) | 15/65 (23.08%) |
| HARD_NEGATIVE (65) | Đỏ | 31/65 (47.69%) | 13/65 (20.00%) |
| HARD_NEGATIVE (65) | Vàng hoặc đỏ | 44/65 (67.69%) | 28/65 (43.08%) |
| POSITIVE (70) | Xanh | 6/70 (8.57%) | 7/70 (10.00%) |
| POSITIVE (70) | Vàng | 10/70 (14.29%) | 5/70 (7.14%) |
| POSITIVE (70) | Đỏ | 54/70 (77.14%) | 58/70 (82.86%) |
| POSITIVE (70) | Vàng hoặc đỏ | 64/70 (91.43%) | 63/70 (90.00%) |

## Chỉ số tổng hợp

| Chỉ số | XLM-R chỉ học risk | Public |
|---|---:|---:|
| Strict accuracy | 57.00% | 76.50% |
| Macro-F1 hai nhãn có gold (cách README public) | 64.57% | 81.79% |
| Macro-recall hai nhãn có gold | 61.65% | 77.97% |
| Operational accuracy (cách public) | 68.50% | 86.50% |
| Recall cảnh báo vàng/đỏ trên POSITIVE | 91.43% | 90.00% |
| Macro-F1 đủ ba nhãn | 43.05% | 54.53% |
| Weighted-F1 | 63.57% | 81.82% |

## Precision, recall và F1 theo lớp

| Lớp | Chỉ số | XLM-R chỉ học risk | Public |
|---|---|---:|---:|
| NO_EVIDENCE | precision | 90.91% | 93.14% |
| NO_EVIDENCE | recall | 46.15% | 73.08% |
| NO_EVIDENCE | f1-score | 61.22% | 81.90% |
| HIGH_EVIDENCE | precision | 60.67% | 80.56% |
| HIGH_EVIDENCE | recall | 77.14% | 82.86% |
| HIGH_EVIDENCE | f1-score | 67.92% | 81.69% |

## Đúng/sai theo cách chấm public

| Nhóm | Cách chấm | XLM-R chỉ học risk | Public |
|---|---|---:|---:|
| EASY_NEGATIVE | Strict | 39/65 (60.00%) | 58/65 (89.23%) |
| EASY_NEGATIVE | Operational | 39/65 (60.00%) | 58/65 (89.23%) |
| HARD_NEGATIVE | Strict | 21/65 (32.31%) | 37/65 (56.92%) |
| HARD_NEGATIVE | Operational | 34/65 (52.31%) | 52/65 (80.00%) |
| POSITIVE | Strict | 54/70 (77.14%) | 58/70 (82.86%) |
| POSITIVE | Operational | 64/70 (91.43%) | 63/70 (90.00%) |

## Tổng số đầu ra (không phải số mẫu gold)

| Đầu ra | XLM-R chỉ học risk | Public |
|---|---:|---:|
| NO_EVIDENCE | 66/200 (33.00%) | 102/200 (51.00%) |
| REVIEW | 45/200 (22.50%) | 26/200 (13.00%) |
| HIGH_EVIDENCE | 89/200 (44.50%) | 72/200 (36.00%) |

## Quy tắc và giới hạn

- Strict yêu cầu nhãn dự đoán trùng gold; mọi REVIEW bị tính sai vì không có gold REVIEW.
- Macro-F1 hai nhãn bỏ REVIEW khỏi phép lấy trung bình, không đổi REVIEW thành HIGH_EVIDENCE.
- Operational public: POSITIVE chấp nhận vàng/đỏ, HARD_NEGATIVE chấp nhận xanh/vàng, EASY_NEGATIVE chỉ chấp nhận xanh. Vì thế số operational không bằng accuracy của phép gộp vàng/đỏ thành một lớp đồng nhất.
- Trên gold NO_EVIDENCE, vàng/đỏ được thống kê như cảnh báo vượt nhãn gold; operational vẫn chấp nhận vàng ở HARD_NEGATIVE. Không gọi tất cả cảnh báo này là cáo buộc lừa đảo.
- Dùng đúng code metric public cho cả hai. Không sửa dự đoán/ngưỡng/model. Một seed, đánh giá lại trên bộ test đã xem. Public kế thừa cả action và semantic pretraining; chưa tách tác dụng riêng của semantic.
- Nguồn: EXACT_PUBLIC_SCORING.json, được tính từ 200 prediction đã khóa của mỗi bản.

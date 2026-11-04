# Baseline XLM-R chỉ học risk

Khởi tạo **XLM-R gốc → train risk**, không nạp RUN004A hay checkpoint semantic; không dùng nhãn action/semantic. Giữ cấu trúc pooling hội thoại, classifier, 3.805 risk TRAIN / 398 DEV và cấu hình giai đoạn risk của model public. Mục tiêu là so sánh toàn bộ pipeline có học bổ trợ action/semantic trước risk với train risk trực tiếp.

Kết quả: [báo cáo đầy đủ](../../evaluation/semantic_pretraining_comparison/README.md). Model baseline được phân phối qua Release riêng, không thay `model-v1`.

## Train lại

Từ gốc repository, dùng môi trường Python theo README chính, có PyTorch CUDA. Chuẩn bị XLM-R base gốc tại `artifacts/base/xlm-roberta-base`:

```powershell
python scripts/prepare_xlmr_base.py
python experiments/xlmr_risk_only/experiment.py train
python experiments/xlmr_risk_only/experiment.py infer
python experiments/xlmr_risk_only/score_exact_public.py
```

Có thể đặt `SCAMGUARD_BASE_MODEL` tới snapshot local. SHA256 trọng số gốc được kiểm tra trước train. Các đầu ra được ghi trong `outputs/` (không commit), không sửa kết quả đã lưu trong `evaluation/`.

3 epoch; seed 11019; AdamW LR 1.5e-5, weight decay 0.01; batch 2, accumulation 8; BF16; cố định embeddings và 6 block dưới; tối đa 192 token/12 lượt; weighted cross-entropy. Chọn theo DEV macro-F1, không theo test. Dùng parser lịch sử của lần đánh giá 200 đã công bố, được chép trực tiếp vào script để không phụ thuộc runtime semantic.

Muốn tải checkpoint thay vì train: `python scripts/download_research_assets.py baseline`. Sau đó chạy `infer` và `score_exact_public.py` nếu chủ động muốn chạy lại; để chỉ kiểm tra báo cáo, dùng `python scripts/verify_research_results.py`.

Đây là một seed và đánh giá so sánh lại trên test đã xem. Public kế thừa cả action RUN004A và semantic pretraining, đồng thời có thêm dữ liệu/thời gian học bổ trợ; kết quả không tách riêng nhân quả của bốn thuộc tính semantic.

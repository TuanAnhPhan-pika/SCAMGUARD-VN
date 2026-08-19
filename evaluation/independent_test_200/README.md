# Independent test 200

Artefact của lần kiểm thử độc lập đã đóng băng ngày 19/08/2026.

- 70 positive, 65 hard negative, 65 easy negative.
- Không có exact overlap với TRAIN/DEV tại thời điểm kiểm tra.
- Model chỉ đọc `FINAL_MODEL_INPUT.jsonl`.
- `predictions_frozen.jsonl` được ghi và hash trước khi mở `INTERNAL_GOLD.jsonl`.
- Gold chỉ có hai nhãn `NO_EVIDENCE` và `HIGH_EVIDENCE`; không có gold `REVIEW`.

Không sửa các file trong thư mục này nếu muốn kiểm chứng kết quả công bố. `infer.py` cố ý từ chối chạy khi prediction đã tồn tại. Dùng `scripts/verify_release.py` để xác minh hash và tính lại metric theo cách chỉ đọc.


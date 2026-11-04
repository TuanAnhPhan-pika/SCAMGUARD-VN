# ScamGuard-VN

ScamGuard-VN là proof-of-concept nhận diện hành động nhạy cảm và mức rủi ro trong tin nhắn/hội thoại tiếng Việt. Hệ thống dùng XLM-R để dự đoán:

- `CandidateAction`: 10 lớp hành động;
- bốn thuộc tính ngữ dụng: `Requested`, `Negated`, `Quoted`, `Reported`;
- mức rủi ro: `NO_EVIDENCE`, `REVIEW`, `HIGH_EVIDENCE`.

Ứng dụng Android nhận văn bản, ảnh/OCR hoặc QR rồi gọi API chạy model trên laptop. Phần sinh hướng dẫn bằng Ollama Cloud là tùy chọn và không tham gia quyết định nhãn.

> Đây là công cụ hỗ trợ nghiên cứu, không phải dịch vụ xác minh danh tính hay kết luận pháp lý/tài chính.

## Mô hình đã huấn luyện

Tải hai checkpoint tại **[Release Model v1](https://github.com/TuanAnhPhan-pika/SCAMGUARD-VN/releases/tag/model-v1)**, trong mục **Assets**:

| File | Chức năng |
|---|---|
| `checkpoint_best.pt` | Nhận diện hành động và bốn thuộc tính ngữ dụng |
| `risk_e2e_best.pt` | Phân loại ba mức rủi ro |

Đặt cả hai file vào `artifacts/checkpoints/`. Các file checkpoint được lưu trong Release, không đi kèm khi clone repository hoặc tải ZIP mã nguồn. Có thể tải tự động bằng `scripts/download_checkpoints.ps1` sau khi cài môi trường theo hướng dẫn bên dưới.

**Không cần huấn luyện lại để chạy thử:** sử dụng hai checkpoint đã cung cấp và làm theo mục “Chạy nhanh trên Windows”.

## Kết quả chính

| Tập đánh giá | Chỉ số | Kết quả |
|---|---:|---:|
| Semantic DEV (486 mẫu) | CandidateAction macro-F1 | 96,14% |
| Semantic DEV | Supported-action macro-F1 | 99,13% |
| Risk DEV (398 mẫu) | Macro-F1 | 96,72% |
| Independent test (200 mẫu) | Strict accuracy | 76,50% |
| Independent test | Macro-F1 trên 2 nhãn có support | 81,79% |
| Independent test | Operational accuracy | 86,50% |
| Independent test | Positive được cảnh báo vàng/đỏ | 90,00% |

Bộ independent test chỉ có gold `NO_EVIDENCE` và `HIGH_EVIDENCE`; vì vậy macro-F1 ba lớp 54,53% được công bố nhưng không dùng làm chỉ số đại diện. Chi tiết ở [`evaluation/independent_test_200`](evaluation/independent_test_200).

## Thử nghiệm bổ sung: baseline và INT8

Hai nghiên cứu bổ sung được giữ riêng; model/API/Android và kết quả independent test gốc không bị thay thế.

| Nội dung | Kết quả chính | Chi tiết |
|---|---|---|
| XLM-R gốc → chỉ train risk, so với pipeline có action/semantic pretraining | Accuracy 57,00% / 76,50%; macro-F1 hai nhãn 64,57% / 81,79%; NO_EVIDENCE đoán đúng 46,15% / 73,08% | [Bảng đầy đủ và cách chấm](evaluation/semantic_pretraining_comparison/README.md) |
| Risk-only INT8 cải thiện, so với ONNX FP32 | 1111,6 → 281,6 MB; accuracy 76,00% → 75,50%; median CPU laptop 37,06 → 16,46 ms | [Các vòng nén, tốc độ và giới hạn](evaluation/risk_int8/README.md) |

Đây là **comparative re-evaluation**, không phải hai test độc lập mới. Baseline dùng prediction gốc và parser lịch sử (public 76,5%); thử nghiệm nén giữ parser hiện tại (FP32 76,0%). Không gộp hai bảng như thể cùng preprocessing. Baseline chưa học action/semantic; INT8 vẫn kế thừa encoder đã học chúng. Một seed baseline chưa đo độ dao động. Hai thư mục `experiments/` có code tái lập; trọng số bổ sung phân phối qua Release thử nghiệm riêng, không ghi đè `model-v1`.

Kiểm tra tất cả kết quả bổ sung chỉ bằng prediction đã lưu, không train/suy luận lại:

```powershell
python scripts/verify_research_results.py
```

## Cấu trúc repository

```text
android/                       ứng dụng Android (OCR, QR, gọi local API)
server/                        FastAPI phục vụ model và guidance tùy chọn
semantic_retrain_10class/      semantic model, runtime và script train
risk_retrain_e2e/              risk model, runtime và script train
runtime/research/v3/jarp_vn/   encoder/collator lõi
data/splits/                   bốn split TRAIN/DEV dùng trong báo cáo
evaluation/independent_test_200/ input, gold, prediction, report và hash
reports/training/              lịch sử metric DEV
experiments/xlmr_risk_only/    baseline XLM-R gốc chỉ học nhãn risk
experiments/risk_int8/         export/nén/runtime INT8 thử nghiệm
evaluation/semantic_pretraining_comparison/  đối chứng baseline/public
evaluation/risk_int8/          các vòng nén, prediction và benchmark CPU
scripts/                       cài đặt, tải checkpoint và xác minh artefact
```

## Chạy nhanh trên Windows

Yêu cầu: Python 3.11/3.12, Git, khoảng 5 GB trống. GPU không bắt buộc cho suy luận; CPU sẽ chậm hơn.

```powershell
git clone https://github.com/TuanAnhPhan-pika/SCAMGUARD-VN.git
cd SCAMGUARD-VN
./scripts/setup_windows.ps1
./scripts/download_checkpoints.ps1
./scripts/run_server.ps1
```

Kiểm tra:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/health
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8765/v1/analyze-action `
  -ContentType 'application/json' -Body '{"text":"Không cung cấp OTP cho bất kỳ ai."}'
```

Checkpoint lần đầu sẽ tải tokenizer `xlm-roberta-base` từ Hugging Face. Muốn chạy hoàn toàn offline, tải model này trước và đặt `SCAMGUARD_BASE_MODEL` tới thư mục local cùng `SCAMGUARD_LOCAL_FILES_ONLY=1`.

## Kết nối Android

1. Mở repository bằng Android Studio và chọn module `android`.
2. Chạy API trên laptop tại cổng `8765`.
3. Kết nối điện thoại qua USB, bật USB debugging và chạy:

```powershell
adb reverse tcp:8765 tcp:8765
```

4. Build/run ứng dụng từ Android Studio. Emulator có thể dùng cùng lệnh `adb reverse`.

Ứng dụng dùng đúng ba nhãn của API công bố: `NO_EVIDENCE` → xanh, `REVIEW` → cam, `HIGH_EVIDENCE` → đỏ. Nhãn thiếu hoặc không hợp lệ được báo là không có kết quả; API lịch sử dùng `SAFE`/`SCAM` không thuộc giao thức của bản ứng dụng này. Khi guidance chưa khả dụng, kết quả phân loại vẫn hiển thị và app báo chưa có hướng dẫn.

Kiểm tra ánh xạ nhãn, xử lý JSON/guidance và luồng HTTP bằng phản hồi mẫu, rồi build APK:

```powershell
.\gradlew.bat :android:testDebugUnitTest :android:assembleDebug
```

Các kiểm tra này không chạy lại mô hình hoặc bộ independent test 200.

## Tái lập kết quả đã công bố

Xác minh hash và tính lại metric từ prediction đã đóng băng:

```powershell
.\.venv\Scripts\python.exe scripts\verify_release.py
```

Script không suy luận lại và không sửa artefact. Quy trình independent test gốc đã khóa checkpoint, input và prediction trước khi mở `INTERNAL_GOLD`; trạng thái cuối là `FINAL_INDEPENDENT_TEST_VALID=YES`.

## Huấn luyện lại từ đầu

Máy huấn luyện gốc dùng RTX 4050 Laptop GPU, batch 2, gradient accumulation 8 và 3 epoch.

```powershell
$env:SCAMGUARD_DATA_SPLITS = "$PWD\data\splits"
.\.venv\Scripts\python.exe -m semantic_retrain_10class.train_semantic --epochs 3
$env:SCAMGUARD_SEMANTIC_CHECKPOINT = "$PWD\artifacts\checkpoints\checkpoint_best.pt"
.\.venv\Scripts\python.exe -m risk_retrain_e2e.train
```

Seed đã được cố định trong code. Kết quả có thể dao động nhẹ do GPU, phiên bản CUDA/PyTorch và kernel không hoàn toàn xác định. Dùng checkpoint trong Release để tái lập suy luận gần nhất với báo cáo.

## Ollama Cloud guidance (tùy chọn)

Không đặt API key trong source hoặc APK. Chỉ cấu hình ở máy chạy server:

```powershell
$env:OLLAMA_API_KEY = "YOUR_KEY"
$env:OLLAMA_GUIDANCE_MODEL = "gemma4:31b"
```

Nếu không có khóa, classifier vẫn chạy; Android chỉ báo phần guidance chưa khả dụng.

## Dữ liệu và giới hạn

Dataset gồm dữ liệu người viết, dữ liệu tổng hợp có kiểm soát, hard negative và các cặp phản thực. Một phần dữ liệu AI vẫn cần human review đầy đủ. Xem [`data/DATASET_CARD.md`](data/DATASET_CARD.md) trước khi sử dụng hoặc công bố lại.

## Trích dẫn

Thông tin trích dẫn chính thức sẽ được bổ sung sau khi công trình được công bố. Hiện có thể dẫn repository và commit hash đã sử dụng.


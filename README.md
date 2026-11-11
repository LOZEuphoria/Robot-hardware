# Robot-hardware

Dự án Python nhận diện mặt sàn bằng PIDNet và tính lệnh điều hướng robot bằng bộ điều khiển PID. Có chế độ mô phỏng từ video, chạy camera trực tiếp, tránh vật cản và gửi lệnh động cơ tới ESP8266 qua serial.

## Cài đặt

Chạy các lệnh tại thư mục gốc dự án. Môi trường Python đi kèm dự án ban đầu dùng Python 3.12; cần tạo lại môi trường trên máy của bạn.

```powershell
git clone --recurse-submodules https://github.com/LOZEuphoria/Robot-hardware.git
cd Robot-hardware
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` lưu các phiên bản phụ thuộc của dự án; việc cài mới chưa được kiểm chứng trên mọi nền tảng. Các chương trình mặc định dùng CUDA nếu PyTorch nhận được GPU tương thích, nếu không sẽ dùng CPU. Lớp `PIDNetRoadSegmenter` cũng hỗ trợ `device="directml"` khi được truyền rõ ràng.

## Tài nguyên cần chuẩn bị

Trọng số mô hình, dữ liệu huấn luyện, môi trường ảo và kết quả sinh ra được lưu riêng, không đi kèm mã nguồn trên GitHub. Video mẫu `7882087290505.mp4` được giữ trong repository.

| Tài nguyên | Đường dẫn sử dụng |
| --- | --- |
| Checkpoint phân đoạn sàn hai lớp (`0`: nền, `1`: sàn) | `checkpoints_floor/pidnet_floor_best.pt` |
| Video đầu vào cho mô phỏng | `7882087290505.mp4`, hoặc sửa `VIDEO_PATH` trong `pidnet_pid_steering.py` |
| Ảnh và nhãn COCO để huấn luyện | `dataset/train/`, `dataset/valid/`, `dataset/test/`; mỗi thư mục chứa ảnh và `_annotations.coco.json` |
| Trọng số Cityscapes tùy chọn để khởi tạo huấn luyện | `PIDNet/pretrained_models/cityscapes/PIDNet_L_Cityscapes_test.pt` |

Checkpoint sàn phải được cung cấp riêng hoặc tạo bằng quy trình huấn luyện bên dưới. Các trọng số Cityscapes gốc và tài liệu PIDNet được giới thiệu trong [README của PIDNet](https://github.com/XuJiacong/PIDNet/blob/4c158cf24ce432f0a8cb43364fae38d93cee0dc3/README.md).

## Chạy camera và mô phỏng

Trước khi chạy camera, chỉnh [config.py](config.py):

- `CAM_ID`: mặc định `0`; ảnh camera `640 × 480`, ảnh đưa vào mô hình `320 × 240`.
- `PIDNET_MODEL_SIZE`: phải khớp với checkpoint sàn. Cấu hình hiện tại là `pidnet-s`, trong khi chương trình huấn luyện và mô phỏng dùng `pidnet-l`; nếu dùng checkpoint từ cấu hình huấn luyện hiện tại, đổi thành `pidnet-l`.
- `SEND_TO_ESP_SERIAL`: hiện là `True`; đặt `False` khi chỉ muốn xem kết quả camera. Cổng mặc định `COM6`, baud rate `115200`, tần số gửi `15 Hz`.
- Các tham số vùng quan sát, tốc độ và tránh vật cản cũng nằm trong file này.

Camera dùng `checkpoints_floor/pidnet_floor_best.pt` được khai báo trong `pidnet_pid_steering_camera.py`; biến `PIDNET_WEIGHT` trong `config.py` không thay đổi đường dẫn checkpoint của chương trình này.

```powershell
# Camera trực tiếp; nhấn q trong cửa sổ hiển thị để thoát
.\.venv\Scripts\python.exe pidnet_pid_steering_camera.py

# Mô phỏng bằng video với checkpoint pidnet-l
.\.venv\Scripts\python.exe pidnet_pid_steering.py
```

Chế độ camera cần camera hoạt động và firmware ESP tương thích nếu bật serial. Lệnh serial là JSON theo từng dòng, gồm `frame_idx`, `steering_value`, `speed_scale`, `left_motor`, `right_motor`, `turn_type` và `motion_mode`. Kết quả được ghi trong `outputs_pidnet/`; video mô phỏng là `outputs_pidnet/steering_simulation.mp4`.

## Huấn luyện mô hình sàn

Chuẩn bị dữ liệu COCO có nhãn polygon tên `floor`, rồi chạy:

```powershell
.\.venv\Scripts\python.exe preprocessing/convert_coco_to_floor_masks.py
.\.venv\Scripts\python.exe preprocessing/check_masks_pil.py
.\.venv\Scripts\python.exe train_pidnet_floor.py
```

Bước chuyển đổi tạo mask nhị phân trong `dataset/masks/{train,valid,test}/` và ảnh xem trước trong `dataset/previews/`. Chương trình huấn luyện dùng tập `train` và `valid`, mặc định `pidnet-l`, kích thước ảnh `512 × 384`, batch size `4`, `50` epoch. Chỉnh các thông số ở đầu `train_pidnet_floor.py` theo dữ liệu và phần cứng.

Nếu thiếu trọng số Cityscapes tùy chọn, chương trình huấn luyện từ đầu. Checkpoint tốt nhất và checkpoint cuối được lưu lần lượt tại `checkpoints_floor/pidnet_floor_best.pt` và `checkpoints_floor/pidnet_floor_last.pt`.

## Nguồn PIDNet

Thư mục `PIDNet/` là Git submodule của [PIDNet do Jiacong Xu và cộng sự phát triển](https://github.com/XuJiacong/PIDNet), tại commit `4c158cf24ce432f0a8cb43364fae38d93cee0dc3`. Nếu đã clone mà chưa tải submodule, chạy `git submodule update --init --recursive`.

Xem `PIDNet/README.md` sau khi clone, hoặc [tài liệu gốc](https://github.com/XuJiacong/PIDNet/blob/4c158cf24ce432f0a8cb43364fae38d93cee0dc3/README.md) và [giấy phép MIT của PIDNet](https://github.com/XuJiacong/PIDNet/blob/4c158cf24ce432f0a8cb43364fae38d93cee0dc3/LICENSE) để biết cách sử dụng, trích dẫn và điều kiện phân phối mã nguồn này.

## HUFLIT RAG

Mã nguồn trợ lý RAG, corpus, bộ lập chỉ mục Elasticsearch và hướng dẫn chạy nằm trong [RAG/README.md](RAG/README.md). Corpus có thêm văn bản trích xuất từ phụ lục kết quả HIRA Paper 315; PDF gốc nằm trong `RAG/documents/`.

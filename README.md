# CVTHead Fine-Tuning Project

## 1. Giới thiệu

Dự án này triển khai và fine-tune mô hình **CVTHead** nhằm tạo **Talking Head Animation** từ một ảnh khuôn mặt duy nhất.
Mô hình có thể điều khiển chuyển động khuôn mặt thông qua các hệ số biểu cảm và pose để sinh ra các animation khác nhau.

Pipeline tổng quát của hệ thống:

```
Input Image
     ↓
Face Landmark Detection
     ↓
DECA Encoder
     ↓
Vertex Feature Transformer
     ↓
Neural Rendering
     ↓
Generated Face Animation
```

---

# 2. Cấu trúc thư mục

```
CVTHead/
│
├── models/              # Kiến trúc mô hình
├── dataset1/            # Dataset loader
├── data/                # Pretrained weights
├── checkpoints/         # Checkpoint sau khi fine-tune
├── examples/            # Ảnh test
├── outputs/             # Kết quả inference
│
├── train.py             # Script huấn luyện
├── inference.py         # Script inference
└── README.md
```

---

# 3. Dataset

Dự án sử dụng dataset **FFHQ (Flickr-Faces-HQ)**.

Dataset gồm khoảng **70.000 ảnh khuôn mặt chất lượng cao**. Trong dự án này, toàn bộ ảnh đã được **resize về 256×256** để phù hợp với mô hình CVTHead và giảm chi phí tính toán khi huấn luyện.

---

# 4. Dataset Processing

Trước khi đưa vào huấn luyện, dữ liệu được tiền xử lý qua các bước sau:

### 1. Face Landmark Detection

Phát hiện các landmark trên khuôn mặt (mắt, mũi, miệng) bằng thư viện:

```
face-alignment
```

### 2. Face Crop

Cắt vùng khuôn mặt từ ảnh gốc và chuẩn hóa kích thước về:

```
256 × 256
```

### 3. Face Alignment

Căn chỉnh khuôn mặt bằng phép biến đổi:

```
Similarity Transform
```

Transform matrix được lưu lại để sử dụng trong quá trình training:

```
src_tform
drv_tform
```

### 4. Normalization

Ảnh được chuyển sang tensor và chuẩn hóa giá trị pixel về:

```
[-1, 1]
```

Ví dụ:

```python
img = (img / 127.5) - 1
```

Sau bước preprocessing, dataset loader cung cấp các dữ liệu:

```
src_img
drv_img
crop_src_img
crop_drv_img
src_tform
drv_tform
```

---

# 5. Quá trình Fine-Tuning

Fine-tuning được thực hiện từ mô hình pretrained của CVTHead.

Load mô hình:

```python
model = CVTHead()
model.load_state_dict(torch.load("data/cvthead.pt"))
```

Huấn luyện mô hình bằng lệnh:

```
python train.py
```

Các checkpoint của mô hình sẽ được lưu trong thư mục:

```
checkpoints/
```

Ví dụ:

```
ckpt_5.pt
ckpt_10.pt
ckpt_20.pt
```

---

# 6. Inference

Sau khi hoàn tất quá trình fine-tuning, mô hình có thể được sử dụng để tạo animation từ một ảnh đầu vào.

Chạy lệnh:

```
python inference.py \
--src_pth examples/1.png \
--ckpt_pth checkpoints/ckpt_20.pt \
--out_dir outputs
```

Trong đó:

* `src_pth`: đường dẫn ảnh khuôn mặt đầu vào
* `ckpt_pth`: checkpoint của mô hình sau khi huấn luyện
* `out_dir`: thư mục lưu kết quả

---

# 7. Output

Sau khi chạy inference, hệ thống sẽ tạo các animation GIF trong thư mục `outputs`:

```
outputs/
│
├── shape.gif
├── exp.gif
├── pose.gif
└── jaw.gif
```

Ý nghĩa các file:

* **shape.gif**: thay đổi hình dạng khuôn mặt
* **exp.gif**: thay đổi biểu cảm khuôn mặt
* **pose.gif**: thay đổi góc quay đầu
* **jaw.gif**: mô phỏng chuyển động hàm

Các animation được tạo bằng cách thay đổi **FLAME coefficients** trong quá trình inference.

---

# 8. Tài liệu tham khảo

1. CVTHead: One-shot Controllable Head Avatar with Vertex-feature Transformer
2. DECA: Detailed Expression Capture and Animation
3. FLAME: Faces Learned with an Articulated Model and Expressions
4. FFHQ Dataset

# Data và Pretrained Models cho CVTHead

Thư mục này chứa các tệp cần thiết để chạy mô hình **CVTHead**, bao gồm các mô hình đã được huấn luyện trước (pretrained models) và các tệp hỗ trợ cho quá trình tái tạo và biến dạng khuôn mặt.

Các tệp này được sử dụng trong các bước **training, fine-tuning và inference** của hệ thống.

---

# 1. Các tệp đã có sẵn trong thư mục

Các tệp sau đã được cung cấp sẵn trong project:

* **mean_deform.pt**
  Mô hình biến dạng dùng cho phần tóc và vai.

* **u_full.pt**
  Thành phần của mô hình biến dạng tóc/vai.

* **transform.pkl**
  Dùng để giảm và tăng số lượng đỉnh của mesh từ **5024 đỉnh xuống 314 đỉnh**.
  File này được sử dụng trong **SpiralNet** để xử lý mesh khuôn mặt.

* **head_template.obj**
  Mô hình khuôn mặt chuẩn của **FLAME face model**.

---

# 2. Các tệp cần tải thêm

Để chạy đầy đủ hệ thống, bạn cần tải các mô hình pretrained sau:

* **cvthead.pth**
  Mô hình CVTHead đã được huấn luyện trước trên dataset **VoxCeleb1**.

* **79999_iter.pth**
  Mô hình **Face Parsing** dùng để phân đoạn khuôn mặt.

* **linear_hair.pth**
  Mô hình biến dạng tóc/vai từ dự án **ROME**.

* **rome.pth**
  Mô hình **ROME pretrained**, được dùng để trích xuất đặc trưng CNN.

* **generic_model.pkl**
  Mô hình **FLAME (Face Model)**.

* **resnet50_scratch_weight.pth**
  Mô hình **ResNet-50 pretrained trên VGGFace2**, dùng để tính **Face Identity Loss**.

* **deca_model.tar**
  Mô hình **DECA pretrained** dùng cho việc ước lượng tham số khuôn mặt 3D.

---

# 3. Tải toàn bộ dữ liệu tự động

Bạn có thể tải tất cả các tệp cần thiết bằng script sau:

```bash
bash fetch_data.sh
```

Script này sẽ tự động tải toàn bộ các mô hình và dữ liệu cần thiết cho project.

---

# 4. Tải thủ công từng mô hình

Nếu script tải tự động không hoạt động, bạn có thể tải từng tệp từ các nguồn chính thức dưới đây.

---

## Face Parsing

File:
79999_iter.pth

Link tải:
https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view

Repository:
https://github.com/VisionSystemsInc/face-parsing.PyTorch

---

## ROME

### Linear Hair Deformation Model

linear_hair.pth

Link tải:
https://drive.google.com/file/d/1Enw9MU9Xin77ws08y4pNqkMW0AyUIzv_/view

### ROME Pretrained Model

rome.pth

Link tải:
https://drive.google.com/file/d/1rLtc037Ra6Z6t0kp-gJ8P1ZKfzkKm070/view

Repository:
https://github.com/SamsungLabs/rome

---

## DECA

Repository chính thức:
https://github.com/yfeng95/DECA

---

## FLAME Face Model

Trang chủ:
https://flame.is.tue.mpg.de/

---

## Face Identity Loss (ResNet50)

File:
resnet50_scratch_weight.pth

Link tải:
https://drive.google.com/file/d/17bGCDQLuXU81xqHF1MB6nBqpBO6PtPd2/view

Repository:
https://github.com/cydonia999/VGGFace2-pytorch

---

# 5. Lưu ý

Sau khi tải xong, hãy đặt các tệp vào đúng thư mục **data/** hoặc thư mục được yêu cầu trong project để đảm bảo mô hình có thể hoạt động bình thường.

Nếu thiếu bất kỳ tệp nào trong danh sách trên, quá trình **training vàerence của CVTHead sẽ không chạy được.**

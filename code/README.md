# Hướng dẫn chạy dự án Học Liên kết với Flower

Dự án này minh họa một hệ thống học liên kết (Federated Learning) đơn giản sử dụng thư viện Flower. Nó bao gồm ba thành phần chính: một server (`server.py`), các client (`client.py`), và một notebook Jupyter (`fl.ipynb`) để mô phỏng một kịch bản học liên kết phức tạp hơn với PyTorch.

## Cấu trúc thư mục

```
.
├── client.py       # Kịch bản để chạy một client
├── server.py       # Kịch bản để chạy server điều phối
└── fl.ipynb        # Notebook mô phỏng học liên kết với PyTorch và CIFAR-10
```

## 1. `server.py` - Server điều phối

File này chịu trách nhiệm khởi tạo và chạy server trung tâm, điều phối quá trình huấn luyện và đánh giá trên các client.

### Chức năng chính:

- **Khởi tạo Strategy**: Sử dụng `FedAvg` (Federated Averaging) làm chiến lược. Server sẽ tính trung bình có trọng số các cập nhật từ client để tạo ra mô hình toàn cục mới.
- **Cấu hình các hàm callback**:
  - `on_fit_config_fn`: Gửi thông tin cấu hình (ví dụ: số vòng hiện tại) đến các client trước khi huấn luyện.
  - `on_evaluate_config_fn`: Gửi thông tin cấu hình đến client trước khi đánh giá.
  - `evaluate_metrics_aggregation_fn`: Tùy chỉnh cách server tổng hợp các chỉ số (metrics) từ client. Ở đây, nó tính trung bình có trọng số của `r2_score`.
  - `evaluate_fn`: Thực hiện đánh giá mô hình toàn cục trên một tập dữ liệu phía server. Điều này hữu ích để có một cái nhìn khách quan về hiệu suất của mô hình.
- **Khởi động Server**: `fl.server.start_server` lắng nghe các kết nối từ client tại địa chỉ `0.0.0.0:8080` và chạy trong 5 vòng (`num_rounds=5`).

### Cách chạy:

Mở một terminal và chạy lệnh sau:

```bash
python server.py
```

Server sẽ khởi động và chờ các client kết nối.

## 2. `client.py` - Client tham gia huấn luyện

File này định nghĩa logic cho mỗi client tham gia vào quá trình học liên kết.

### Chức năng chính:

- **Tải và phân chia dữ liệu**: Mỗi client sẽ tải một phần dữ liệu huấn luyện riêng biệt. Trong ví dụ này, dữ liệu được tạo giả lập và chia đều cho hai client dựa trên `client_id` được truyền vào qua dòng lệnh.
- **Định nghĩa `FlowerClient`**:
  - Kế thừa từ `fl.client.NumPyClient` để đơn giản hóa việc xử lý tham số mô hình.
  - `get_parameters()`: Trả về các tham số (trọng số) của mô hình cục bộ.
  - `set_parameters()`: Cập nhật mô hình cục bộ với các tham số nhận được từ server.
  - `fit()`: Huấn luyện mô hình trên dữ liệu cục bộ (`X_train_client`, `y_train_client`) và trả về các tham số đã cập nhật.
  - `evaluate()`: Đánh giá mô hình trên tập dữ liệu kiểm thử chung và trả về các chỉ số hiệu suất (MSE, R2 score).
- **Khởi động Client**: `fl.client.start_numpy_client` kết nối client đến địa chỉ của server.

### Cách chạy:

Bạn cần chạy ít nhất hai client để server có thể bắt đầu quá trình huấn luyện (`min_available_clients=2`).

Mở hai terminal riêng biệt và chạy các lệnh sau:

**Terminal 1 (Client 1):**

```bash
python client.py 1
```

**Terminal 2 (Client 2):**

```bash
python client.py 2
```

Các client sẽ kết nối đến server và bắt đầu quá trình huấn luyện và đánh giá.

## 3. `fl.ipynb` - Mô phỏng Học Liên kết với PyTorch

Đây là một Jupyter Notebook sử dụng tính năng mô phỏng của Flower để thực hiện một kịch bản học liên kết hoàn chỉnh trên bộ dữ liệu CIFAR-10 với mô hình CNN được xây dựng bằng PyTorch.

### Các bước chính trong Notebook:

1.  **Cài đặt thư viện**: Cài đặt `flwr[simulation]`, `flwr-datasets`, `torch`, `torchvision`, và `matplotlib`.
2.  **Tải và phân chia dữ liệu**: Sử dụng `FederatedDataset` của Flower để tự động tải và chia bộ dữ liệu CIFAR-10 cho 10 client mô phỏng.
3.  **Định nghĩa mô hình**: Xây dựng một mô hình CNN đơn giản bằng PyTorch.
4.  **Định nghĩa `FlowerClient`**: Tương tự như `client.py`, nhưng logic `fit` và `evaluate` sử dụng các hàm `train` và `test` của PyTorch.
5.  **Cấu hình và chạy mô phỏng**:
    -   Sử dụng `client_fn` và `server_fn` để tạo client và server một cách linh hoạt.
    -   Cấu hình chiến lược `FedAvg`.
    -   Sử dụng `run_simulation` để khởi chạy toàn bộ quá trình mô phỏng mà không cần chạy các kịch bản client/server riêng biệt.

### Cách chạy:

1.  Đảm bảo bạn đã cài đặt Jupyter Notebook hoặc Jupyter Lab.
2.  Mở `fl.ipynb` trong VS Code hoặc môi trường Jupyter của bạn.
3.  Chạy tuần tự các ô (cell) trong notebook để xem toàn bộ quá trình từ chuẩn bị dữ liệu, định nghĩa mô hình, đến huấn luyện và đánh giá liên kết.

---

Bằng cách chạy các thành phần này, bạn sẽ có được một cái nhìn tổng quan về hai cách triển khai một hệ thống học liên kết với Flower:
- **Triển khai thực tế**: Chạy `server.py` và nhiều `client.py` trên các tiến trình hoặc máy khác nhau.
- **Mô phỏng**: Sử dụng `fl.ipynb` để nhanh chóng thử nghiệm và gỡ lỗi các chiến lược và cấu hình khác nhau trong một môi trường được kiểm soát.

PHÂN LOẠI ẢNH ĐA LỚP VỚI DEEP LEARNING

CS406 - HỒ PHẠM QUỐC BẢO - 24520152

GITHUB REPO: https://github.com/24520152-ben/image-classification.git

DÙNG TRỰC TIẾP TRÊN STREAMLIT CLOUD: https://image-classification-ben.streamlit.app/

TÍNH NĂNG CHÍNH
- UPLOAD ẢNH TỪ MÁY NGƯỜI DÙNG
- DỰ ĐOÁN NHÃN CỦA ẢNH (6 NHÃN):
    + buildings
    + forest
    + glacier
    + mountain
    + sea
    + street
- ĐƯA RA CÁC NHÃN DỰ ĐOÁN, THỜI GIAN SUY LUẬN, ĐỘ TỰ TIN CỦA TỪNG MODEL

CẤU TRÚC THƯ MỤC
- checkpoints: thư mục chứa các file keras của model sau khi train và finetune
- dataset
    + test: thư mục chứa ảnh thuộc 6 lớp để test
    + train: thư mục chứa ảnh thuộc 6 lớp để train
- history: thư mục chứa các file csv ghi lại quá trình train và finetune của các mô hình
- src
    + data_loader.py: file tạo dataset
    + model_builders.py: file định nghĩa model
    + feature_extraction.ipynb: notebook để train trước khi finetune
    + fine_tune.ipynb: notebook finetune các model
    + evaluate.ipynb: notebook đánh giá các model
    + app.py: file xây dựng ứng dụng web bằng streamlit
- requirements.txt: file chứa các thư viện cần thiết để streamlit app chạy
- README.md: file hướng dẫn

CÁCH CHẠY LOCAL
- Cài thư viện cần thiết: pip install -r requirements.txt
- Chạy ứng dụng: streamlit run src/app.py
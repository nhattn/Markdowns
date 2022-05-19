# Phân cụm từ tiếng Việt cho máy học

Bài toán phân cụm từ được nghiên cứu và được sử dụng trong nhiều ứng dụng
thực tế như các hệ thống chích trọn thông tin, dịch máy, và tóm tắt văn bản.
Bài toán phân cụm có thể hiểu là việc gộp một dãy liên tiếp các từ trong
câu để gán nhãn cú pháp.

[Kết quả đã được công bố ở SIGNL2001](https://www.clips.uantwerpen.be/conll2000/chunking/)

Quan sát các tập nhãn này chúng ta thấy rằng chúng hoàn toàn tương đồng với
các khái niệm về tập nhãn trong tiếng Việt. Thêm nữa, hầu hết các ứng dụng
như dịch máy, tóm tắt văn bản, trích lọc thông tin đều chủ yếu sự dụng các
loại nhãn này. Điều này hoàn toàn phù hợp với nhu cầu sử dụng các thông tin
về ngữ pháp trong các sản phẩm ứng dụng tiếng Việt đòi hỏi tốc độ nhanh.

Qua khảo sát ngôn ngữ tiếng Việt, chúng ta xác định những tập nhãn cho việc
phân cụm là hữu ích đối với bài toán này. Chúng ta chỉ đưa ra những tập nhãn
chuẩn và xuất hiện nhiều trong câu văn tiếng Việt. Từ đó, chúng tôi đưa ra
bộ nhãn của việc phân cụm từ tiếng Việt bao gồm như sau:

| Tên | Chú thích |
|---|---|
| NP | Cụm danh từ |
| VP | Cụm động từ |
| AP | Cụm tính từ |
| RP | Cụm phó từ |
| PP | Cụm giới từ |
| QP | Cụm từ chỉ số lượng |
| WN | Cụm danh từ nghi vấn (ai, cái gì, con gì, v.v.) |
| WA | Cụm tính từ nghi vấn (lạnh thế nào, đẹp ra sao, v.v.) |
| WR | Cụm từ nghi vấn dùng khi hỏi về thời gian, nơi chốn, v.v.  |
| WP | Cụm giới từ nghi vấn (với ai, bằng cách nào, v.v.) |

1. Ký hiệu: `NP`
    Cấu trúc cơ bản của một cụm danh từ như sau

    ```
    <phần phụ trước> <danh từ trung tâm> <phần phụ sau> 
    ```

    Ví dụ: "**mái tóc đẹp**" thì danh từ "**tóc**" là _phần trung tâm_, định từ
    "**mái**" là _phần phụ trước_, còn tính từ "**đẹp**" là _phần phụ sau_.

    ```
    (NP (D mái) (N tóc) (J đẹp))
    ```
2. Ký hiệu: `VP`
    Giống như cụm danh từ, cấu tạo một cụm động từ về cơ bản như sau:

    ```
    <bổ ngữ trước> <động từ trung tâm> <bổ ngữ sau>
    ```

    Phần phụ trước của cụm động từ thường là phụ từ. 
    Ví dụ: "đang ăn cơm"
    
    ```
    (VP (R đang) (V ăn) (NP cơm)) 
    ```

3. Ký hiệu: `AP`
    Cấu tạo một cụm tính từ về cơ bản như sau:

    ```
    <bổ ngữ trước> <tính từ trung tâm> <bổ ngữ sau>
    ```

    Bổ ngữ trước của tính từ thường là phụ từ chỉ mức độ. 
    Ví dụ: rất đẹp
    
    ```
    (AP (R rất) (J đẹp))
    ```

4. Ký hiệu: `PP`
    Cấu tạo cơ bản như sau:

    ```
    <giới từ> <cụm danh từ>
    ```

    Ví dụ: vào Sài Gòn
    
    ```
    (PP (S vào) (NP Sài Gòn))
    ```
5. Ký hiệu: `QP`
    Thành phần chính của QP là các số từ. Có thể là số từ xác định, số từ
    không xác định, hay phân số. Ngoài ra còn có thể có phụ từ như "**khoảng**",
    "**hơn**", v.v. QP đóng vai trò là thành phần phụ trước trong cụm danh từ
    (vị trí -2). 

    Ví dụ 1: năm trăm
    
    ```
    (QP (M năm) (M trăm)) 
    ```

    Ví dụ 2: hơn 200 
    
    ```
    (QP (R hơn) (M 200)) 
    ```

Phương pháp học nửa giám sát được thực hiện bằng cách hết sức đơn giản. Gồm
các bước sau đây:

> Bước 1: Tạo bộ dữ liệu huấn luyện bé. Bước này được thực hiện bằng việc
nhập liệu từ những câu đã được gắn chuẩn
>
> Bước 2: Sử dụng mô hình CRFs để huấn luyện trên tập dữ liệu này.
>
> Bước 3: Cho tập test và sự dụng CRFs để gán nhãn
>
> Bước 4: Tạo bộ dữ liệu mới. Bộ dữ liệu mới được bổ sung kết quả từ việc
gán nhãn tập test

Để chứng tỏ sự hiểu quả của các phương pháp, chúng ta chia ngẫu nhiên 215
câu làm dữ liệu huấn luyện và 45 câu được sử dụng như dữ liệu để đánh giá
độ chính xác của chương trình.

Sau 45 vòng lặp mô hình CRFs cho kết quả hội tụ. Chúng ta bước đầu đánh giá
độ chính xác của phương pháp phân cụm đối với 45 câu khi thử nghiệm trên
mô hình dùng 215 câu làm dữ liệu huấn luyện.

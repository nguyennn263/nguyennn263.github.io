---
title: Markov chain 
date: 2024-07-05
categories: [Knowledge]
tags: [machine_learning, probability, markov_chain, markov_process]
description:  Markov chain 
media_subpath: /assets/markov-chain/
image:
    path: alt-img.png
---

### Nội dung
- [1. Phát biểu bài toán](#-bai-toan)
- [2. Định nghĩa xích Markov](#-dinh-nghia)
- [3. Ma trận chuyển trạng thái và lược đồ](#-ma-tran-chuyen-trang-thai-va-luoc-do)
- [4. Phân phối xác xuất](#-phan-phoi-xac-xuat)
- [5. Minh hoạ 1 quá trình Markov](#-minh-hoa)
- [6. Tổng kết](#-tong-ket)

Trong bài viết này, ta sẽ tìm hiểu về xích Markov (Markov chain).

<a name="-bai-toan"></a>

## 1. Phát biểu bài toán
Trong một khu vực 1000 người, có 2 cửa hàng bán cùng bán một loại sản phẩm. Người ta thống kê được số liệu ban đầu như sau: 800 người mua hàng ở cửa hàng A, 200 người mua hàng ở cửa hàng B. Sau 1 thời gian cạnh tranh, quảng cáo, khuyến mại,... để thu hút khách hàng của 2 cửa hàng, người ta thống kê lại thì thấy xuất hiện những tình huống sau:
- Xác suất một người trước mua ở cửa hàng A, nay vẫn mua ở của hàng A là $p_{11}$
- Xác suất một người trước mua ở cửa hàng A, nay vẫn mua ở cửa hàng A là $p_{12}$
- Xác suất một người trước mua ở cửa hàng B, nay mua ở cửa hàng A là $p_{21}$
- Xác suất một người trước mua ở cửa hàng B, nay vẫn mua ở cửa hàng B là $p_{22}$


**Câu hỏi đặt ra**: Sau 3 tháng, 6 tháng,..., $n$ tháng nữa thì tỉ lệ số người mua hàng ở mỗi cửa hàng là bao nhiêu?

![Sự thay đổi trạng thái theo thời gian(tương tự với S1)](phat-bieu.png)
_Sự thay đổi trạng thái theo thời gian(tương tự với S1)_

![Lược đồ minh hoạ](minh-hoa.png)
_Lược đồ minh hoạ_

<a name="-dinh-nghia"></a>

## 2. Định nghĩa xích Markov
### 2.1. Không gian trạng thái


Cho tiến trình ngẫu nhiên $\set{X_n, n = 0, 1, 2, ...}$, tập hợp $S$ gồm tất cả các giá trị của dãy biến ngẫu nhiên $(X_n, n = 0, 1, 2,...)$ được gọi là **không gian trạng thái** của tiến trình ngẫu nhiên $(X_n)_n$.

- Ví dụ:
    - Bài toán thói quen mua hàng: $S = \set{0, 1}$
    - Giả sử thời tiết trong ngày chỉ có thể là nắng hoặc mưa: $S = \set{R(rainy), S(sunny)}$
    - $X(t)$ là số người xếp hàng để rút tiền vào thời điểm t, khi đó tập hợp không gian trạng thái của tiến trình ngẫu nhiên ${X_t}$ là: $S = \set{0, 1, 2,...}$

### 2.2. Xích Markov
Xét một tiến trình ngẫu nhiên $\set{X_n, n = 0, 1, 2,...}$ và không gian trạng thái $S \subset \set{0, 1, 2,...}$. Một tiến trình ngẫu nhiên là một xích Markov nếu:
$$
\begin{equation} \notag
P(X_{n+1} = j|X_n = i, X_{n−1} = i_{n−1}, . . . , X_0 = i_0) = P(X_{n+1} = j|X_n = i), \forall n, i, j, i_0, i_1, . . . , i_{n−1}
\end{equation}
$$

**Lưu ý**
- Nếu tập $S$ là hữu hạn thì ta có một xích Markov hữu hạn
- Đặt $p_{ij} = P(X_{n+1} = j \| X_n = i)$, khi đó $p_{ij}$ không phụ thuộc vào $n$, cụ thể:
$$
\begin{equation} \notag
p_{ij} = P(X_1 = j|X_0 = i) = P(X_2 = j|X_1 = i) = P(X_3 = j|X_2 = i) = ...
\end{equation}
$$

<a name="-ma-tran-chuyen-trang-thai-va-luoc-do"></a>

## 3. Ma trận chuyển trạng thái và lược đồ
Giả sử tập trạng thái $S = \set{1, 2, 3, . . . ,r}$, $p_{ij} = P(X_{n+1} = j \| X_n = i)$, khi đó ma trận:

$$
\ {P} = \begin{pmatrix}
p_{11} & p_{12} & \cdots & p_{1r} \\
p_{21} & p_{22} & \cdots & p_{2r} \\
\vdots & \vdots & \ddots & \vdots \\
p_{r1} & p_{r2} & \cdots & p_{rr}
\end{pmatrix}
$$

được gọi là ma trận xác suất chuyển sau **một** bước.

**Lưu ý** 
- Ta luôn có $p_{ij} >= 0$ với mọi $i$ và $\sum_{k = 1}^{r} p_{ik} = \sum_{k = 1}^{r} P(X_{n+1} = k\|X_n = i) = 1$ 

<a name="-phan-phoi-xac-xuat"></a>

## 4. Phân phối xác xuất trạng thái
### 4.1. Xác suất chuyển sau n bước
Xác suất chuyển sau n bước được định nghĩa:
$$
\begin{equation} \notag
p_{ij}^{n} = P(X_{n+m} = j|X_m = i) = P(X_n = j|X_0 = i)
\end{equation}
$$

với quy ước 
$$p_{ij}^{(0)} = 
\begin{cases} 
0 & \text{nếu } i \ne j \\ 
1 & \text{nếu } i = j  \\
\end{cases}
$$
### 4.2. Phân phối xác suất trạng thái
Giả sử có không gian trạng thái $S = \set{1, 2, 3,... ,r}$, phân phối của biến $X_n$ được cho bởi:
$$
\begin{equation} \notag
\pi^{(n)} =  [P(X_n = 1)\quad P(X_n = 2) \quad··· \quad P(X_n = r)]
\end{equation} 
$$

Phân phối $\pi^{(0)}$ của biến $X_0$ được gọi là phân phối ban đầu của hệ.
### 4.3. Định lí
Cho xích Markov $\set{X_n, n = 0, 1, 2,...}$ với ma trận xác xuất chuyển sau một bước là $P$, khi đó:

$$
\begin{equation}  \notag
\pi^{(n+1)} = pi^{(n+1)}P,  \text{với } n = 0, 1, 2,...
\end{equation} 
$$

$$
\begin{equation} \notag
\pi^{(n)} =  pi^{(0)}P^n,  \text{với } n = 0, 1, 2,...
\end{equation} 
$$


<a name="-minh-hoa"></a>

## 5. Minh hoạ 1 quá trình Markov
Bài toán mở đầu chính là bài toán mô hình phân chia thị trường với số lượng cửa hàng là 2. 

Mở rộng: Có $N$ của hàng cùng bán một sản phẩm nào đó. Khách hàng có thể mua hàng ở $N$ của hàng này, việc chọn cửa hàng để mua là tùy thuộc vào sở thích của cửa hàng và họ có thể bỏ cửa hàng này đến một cửa hàng khác. Các cửa hàng sẽ dùng các hình thức cạnh tranh như quảng cáo, khuyến mại,... để lôi kéo khách hàng.

Ví dụ: Có ba cửa hàng và 1000 khách hàng với phân phối ban đầu ở tháng một ($X_0$) như sau:
$$
\begin{equation} \notag
[P(X_0 = 1) = 20\% \quad P(X_0 = 2) = 50\% \quad P(X_0 = 3) = 30\%]
\end{equation} 
$$

Sau 1 tháng (một chu kì thời gian), người ta thu thập được số liệu về việc mua hàng của từng người(mua sản phẩm ở cửa hàng nào). Có được bảng sau:

| Cửa hàng  |Tháng 1 | Từ A sang | Từ B sang | Từ C sang | Khách sang A | Khách sang B | Khách sang C | Tháng 2 |
|-----------|----------------|----------|----------|----------|-----------|-----------|-----------|---------------|
| Cửa hàng A| 200            | 0        | 35       | 25      | 0         | 20        | 20        | 220            |
| Cửa hàng B| 500            | 20       | 0       | 20       | 35         | 0        | 15        | 490           |
| Cửa hàng C| 300            | 20       | 15       | 40       | 25        | 20        | 0        | 290           |

Ta lập được ma trận xác suất chuyển sau:
$$
\ {P} = \begin{pmatrix}
p_{11} = \frac{160}{200} = 0.800 & p_{12} = \frac{20}{200} = 0.100 & p_{13} = \frac{20}{200} = 0.100\\
p_{21} = \frac{35}{500} = 0.070 & p_{22} = \frac{450}{500} = 0.900 & p_{23} = \frac{15}{500} = 0.030 \\
p_{31} = \frac{25}{300} = 0.083 & p_{32} = \frac{20}{300} = 0.067 & p_{33} = \frac{255}{300} = 0.850 
\end{pmatrix}
$$

Dự báo phân chia thị trường cho tương lai:
- Trong tháng Tư thì thị trường sẽ như thế nào? hay phân phối xác suất của từng cửa hàng là bao nhiêu?
- Một năm sau thị trường như thế nào?...

<a name="-tong-ket"></a>

## 6. Tổng kết
Bài viết trên chỉ nói sơ lược về Markov chain. Ở bài viết sau, chúng ta sẽ nói rõ thêm và có 1 vài bài tập nhỏ code bằng python.

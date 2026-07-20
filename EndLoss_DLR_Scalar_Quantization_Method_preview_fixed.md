# End-Loss DLR Scalar Quantization  
## Parallel MM Assignment + Exact Codebook Update

> **Mục tiêu của tài liệu:** mô tả đầy đủ một phương pháp **scalar quantization** cho weight-only quantization, đủ chi tiết để triển khai bằng PyTorch/Triton trên GPU.  
> Mỗi weight vô hướng được thay bằng một trong $K$ **codeword vô hướng**. Phương pháp không phải vector quantization.

---

## 1. Tổng quan phương pháp

Xét một quantization group gồm $n$ scalar weights:

$$
w=(w_1,\ldots,w_n)\in\mathbb R^n.
$$

Ta cần học:

- một scalar codebook

$$
\mathcal C=\{c_1,\ldots,c_K\},\qquad c_k\in\mathbb R;
$$

- một label cho mỗi weight

$$
a_i\in\{1,\ldots,K\}.
$$

Weight sau lượng tử hóa là:

$$
q_i=c_{a_i}.
$$

Quantization error:

$$
e=q-w.
$$

Phương pháp tối ưu luân phiên hai biến trên **cùng một end-loss surrogate**:

1. **Assignment update:** giữ codebook cố định, cập nhật đồng thời label của mọi weight bằng một bước MM song song.
2. **Codebook update:** giữ labels cố định, giải chính xác toàn bộ $K$ codewords bằng một hệ tuyến tính nhỏ kích thước $r\times r$.

Pipeline:

```text
End-loss statistics
        ↓
DLR objective: diagonal + low rank
        ↓
DLR-aware initialization
        ↓
repeat:
    Parallel MM assignment update
    Exact DLR codebook update
until convergence
```

---

# Phần I — Từ end loss đến DLR surrogate

## 2. End-loss objective ban đầu

Gọi:

- $\theta$: toàn bộ tham số của mô hình gốc;
- $\hat\theta=\theta+\Delta\theta$: mô hình sau lượng tử hóa;
- $p_\theta(y\mid x)$: phân phối output của mô hình gốc;
- $p_{\hat\theta}(y\mid x)$: phân phối output của mô hình lượng tử.

Ta dùng mixed end loss:

$$
\mathcal L_{\mathrm{mix}}(\hat\theta)
=
(1-\beta)\,
\mathcal L_{\mathrm{KL}}(\hat\theta;\theta)
+
\beta\,
\left[
\mathcal L_{\mathrm{NLL}}(\hat\theta)
-
\mathcal L_{\mathrm{NLL}}(\theta)
\right].
$$

Trong đó:

$$
\mathcal L_{\mathrm{KL}}(\hat\theta;\theta)
=
\mathbb E_x
D_{\mathrm{KL}}
\left(
p_\theta(\cdot\mid x)
\;\|\;
p_{\hat\theta}(\cdot\mid x)
\right).
$$

Mặc định:

$$
\boxed{\beta=0.5.}
$$

Ý nghĩa:

- KL giữ output của mô hình lượng tử gần mô hình gốc;
- NLL trực tiếp hướng mô hình về task likelihood;
- $\beta=0.5$ cân bằng hai tín hiệu.

---

## 3. Xấp xỉ bậc hai quanh mô hình gốc

Tại $\hat\theta=\theta$:

- teacher KL bằng 0;
- gradient của teacher KL bằng 0;
- NLL có gradient nói chung khác 0.

Dưới Gauss–Newton/Fisher approximation, hai thành phần dùng cùng một output-space curvature $H$. Khi chỉ xét perturbation $e$ của một quantization group, ta nhận được:

$$
\mathcal J(e)
=
\beta g^\top e
+
\frac12 e^\top H e.
$$

Trong đó:

- $g\in\mathbb R^n$: gradient end NLL theo các weights trong group tại mô hình gốc;
- $H\in\mathbb R^{n\times n}$: PSD Gauss–Newton/Fisher curvature của end loss;
- $e=q-w$.

Hằng số không phụ thuộc $q$ đã được bỏ.

---

## 4. Diagonal-plus-low-rank approximation

Không thể lưu hoặc tối ưu trực tiếp với $H$ dense. Ta xấp xỉ:

$$
\boxed{
H\approx D+UU^\top
}
$$

với:

$$
D=\operatorname{diag}(d_1,\ldots,d_n),\qquad d_i>0,
$$

và:

$$
U\in\mathbb R^{n\times r},
$$

trong đó $r$ nhỏ, ví dụ:

$$
r=4.
$$

Loss thực tế của một group là:

$$
\boxed{
\mathcal J(q)
=
\beta g^\top(q-w)
+
\frac12\sum_{i=1}^{n}d_i(q_i-w_i)^2
+
\frac12\left\|U^\top(q-w)\right\|^2.
}
$$

Đây là **objective duy nhất** dùng cho:

- initialization;
- assignment update;
- codebook update;
- stopping criterion.

### 4.1. Tránh double-counting diagonal

Nếu $U$ được xây từ một low-rank approximation của $H$, cần bảo đảm diagonal của $H$ không bị tính hai lần.

Nếu có estimate $\operatorname{diag}(H)$, dùng:

$$
d_i
=
\max\left(
H_{ii}-\|U_{i,:}\|^2,\;
\varepsilon_d
\right).
$$

Khuyến nghị:

```python
d = (diag_H - U.square().sum(dim=-1)).clamp_min(d_min)
```

với `d_min` nhỏ nhưng dương, tính trong FP32.

### 4.2. Input mà solver cần

Mỗi quantization group cần:

```text
w : [n]       original scalar weights
g : [n]       end-NLL gradient
d : [n]       positive diagonal curvature residual
U : [n, r]    low-rank curvature factor
K : int       number of scalar codewords
beta : float  default 0.5
```

Phần ước lượng $g$, $\operatorname{diag}(H)$, và $U$ từ calibration data có thể được triển khai độc lập. Solver bên dưới chỉ giả sử các thống kê này đã có.

---

# Phần II — Biến tối ưu

## 5. Codebook, labels và quantized weights

Codebook:

$$
c=(c_1,\ldots,c_K)\in\mathbb R^K.
$$

Labels:

$$
a=(a_1,\ldots,a_n),\qquad a_i\in\{1,\ldots,K\}.
$$

Quantized weights:

$$
q_i=c_{a_i}.
$$

Có thể viết:

```python
q = codebook[labels]
```

Quantization error:

```python
e = q - w
```

Low-rank state:

$$
h=U^\top e\in\mathbb R^r.
$$

Nhờ $h$, exact loss được tính mà không cần materialize $UU^\top$:

$$
\boxed{
\mathcal J
=
\beta g^\top e
+
\frac12\sum_i d_i e_i^2
+
\frac12\|h\|^2.
}
$$

PyTorch:

```python
def dlr_loss(w, g, d, U, codebook, labels, beta):
    q = codebook[labels]
    e = q - w
    h = U.transpose(-1, -2) @ e
    return (
        beta * torch.dot(g, e)
        + 0.5 * torch.dot(d, e.square())
        + 0.5 * torch.dot(h, h)
    )
```

---

# Phần III — Initialization

## 6. Mục tiêu của initialization

Initialization chỉ cần cung cấp:

- labels ban đầu $a^{(0)}$;
- codebook ban đầu $c^{(0)}$.

Không dùng một solver phụ như GreedySplit hoặc K-means. Initialization phải:

- deterministic;
- nhanh;
- dùng cùng DLR objective;
- tránh cluster rỗng ban đầu.

---

## 7. Continuous DLR optimum

Trước hết bỏ ràng buộc codebook và cho $q\in\mathbb R^n$ tự do.

Loss:

$$
\mathcal J(q)
=
\beta g^\top(q-w)
+
\frac12(q-w)^\top(D+UU^\top)(q-w).
$$

Đặt gradient bằng 0:

$$
\beta g+(D+UU^\top)(q-w)=0.
$$

Continuous optimum:

$$
\boxed{
x
=
w-\beta(D+UU^\top)^{-1}g.
}
$$

$x_i$ trả lời câu hỏi:

> Nếu weight $i$ chưa bị ép vào một codebook rời rạc, end-loss surrogate muốn nó nằm ở đâu?

Trong initialization, $x$ chỉ dùng để:

- sort weights;
- tạo labels ban đầu.

Ta không lượng tử hóa trực tiếp $x$.

---

## 8. Tính $x$ bằng Woodbury

Không nghịch đảo ma trận $n\times n$.

Woodbury:

$$
(D+UU^\top)^{-1}
=
D^{-1}
-
D^{-1}U
\left(I_r+U^\top D^{-1}U\right)^{-1}
U^\top D^{-1}.
$$

Đặt:

$$
y=D^{-1}g,
$$

$$
R=I_r+U^\top D^{-1}U,
$$

$$
v=R^{-1}U^\top y.
$$

Khi đó:

$$
(D+UU^\top)^{-1}g
=
y-D^{-1}Uv.
$$

Cuối cùng:

$$
\boxed{
x=w-\beta\left(y-D^{-1}Uv\right).
}
$$

Pseudocode:

```python
def continuous_dlr_target(w, g, d, U, beta):
    inv_d = 1.0 / d

    # y = D^{-1} g
    y = inv_d * g

    # R = I + U^T D^{-1} U
    DU = inv_d[:, None] * U
    R = torch.eye(U.shape[-1], device=U.device, dtype=U.dtype)
    R = R + U.transpose(-1, -2) @ DU

    # v = R^{-1} U^T y
    rhs = U.transpose(-1, -2) @ y
    v = torch.linalg.solve(R, rhs)

    Hinv_g = y - DU @ v
    return w - beta * Hinv_g
```

Complexity:

$$
O(nr^2+r^3).
$$

---

## 9. Curvature-balanced quantile labels

Tính diagonal sensitivity thật của $D+UU^\top$:

$$
\boxed{
\rho_i=d_i+\|U_{i,:}\|^2.
}
$$

Sort các indices theo $x$:

$$
x_{\pi_1}\le\cdots\le x_{\pi_n}.
$$

Tính cumulative curvature mass:

$$
S_j=\sum_{\ell=1}^{j}\rho_{\pi_\ell}.
$$

Chia dãy thành $K$ nhóm sao cho mỗi nhóm chứa xấp xỉ:

$$
\frac{1}{K}\sum_i\rho_i
$$

curvature mass.

Thresholds:

$$
\tau_k
=
\frac{k}{K}
\sum_i\rho_i,
\qquad
k=1,\ldots,K-1.
$$

Đồng thời ép mỗi cluster có ít nhất một phần tử.

Pseudocode mức thuật toán:

```python
def initialize_labels_from_target(x, d, U, K):
    rho = d + U.square().sum(dim=-1)
    order = torch.argsort(x)
    rho_sorted = rho[order]
    cumsum = torch.cumsum(rho_sorted, dim=0)

    total = cumsum[-1]
    thresholds = total * torch.arange(
        1, K, device=x.device, dtype=x.dtype
    ) / K

    boundaries = torch.searchsorted(cumsum, thresholds)

    # Enforce:
    # 1 <= boundary_1 < ... < boundary_{K-1} <= n-1
    boundaries = enforce_nonempty_intervals(boundaries, n=x.numel(), K=K)

    labels_sorted = labels_from_boundaries(boundaries, n=x.numel(), K=K)

    labels = torch.empty_like(labels_sorted)
    labels[order] = labels_sorted
    return labels
```

---

## 10. Initial codebook

Sau khi có labels ban đầu, chạy **exact DLR codebook update** ở Phần V.

Không lấy trực tiếp quantiles hoặc means làm codeword cuối cùng.

Initialization hoàn chỉnh:

```text
x = continuous DLR optimum
        ↓
sort x
        ↓
curvature-balanced K-way partition
        ↓
initial labels
        ↓
exact DLR codebook update
        ↓
initial codebook
```

---

# Phần IV — Parallel MM Assignment Update

## 11. Bài toán assignment

Giữ codebook $c$ cố định.

Ta cần tìm labels mới, tương đương tìm:

$$
q_i\in\{c_1,\ldots,c_K\},
$$

để giảm:

$$
\mathcal J(q)
=
\beta g^\top(q-w)
+
\frac12(q-w)^\top(D+UU^\top)(q-w).
$$

Mỗi weight được phép nhảy tới **bất kỳ codeword**, không chỉ neighborhood.

Ta muốn mọi weight trong group cập nhật song song.

---

## 12. Gradient tại assignment hiện tại

Tại $q^{(t)}$:

$$
e^{(t)}=q^{(t)}-w,
$$

$$
h^{(t)}=U^\top e^{(t)}.
$$

Gradient theo $q$:

$$
\boxed{
s^{(t)}
=
\beta g
+
D e^{(t)}
+
Uh^{(t)}.
}
$$

Theo từng weight:

$$
s_i^{(t)}
=
\beta g_i
+
d_i e_i^{(t)}
+
U_{i,:}^\top h^{(t)}.
$$

---

## 13. Tại sao cần spectral majorization parameter $\lambda$?

Khi thay đổi đồng thời tất cả weights:

$$
q^{\mathrm{new}}
=
q^{(t)}+\Delta,
$$

loss thay đổi chính xác:

$$
\mathcal J(q^{(t)}+\Delta)
=
\mathcal J(q^{(t)})
+
(s^{(t)})^\top\Delta
+
\frac12\Delta^\top D\Delta
+
\frac12\Delta^\top UU^\top\Delta.
$$

Term cuối couple các weights:

$$
\Delta^\top UU^\top\Delta
=
\|U^\top\Delta\|^2.
$$

Ta chọn:

$$
\boxed{
\lambda
\ge
\lambda_{\max}(UU^\top)
=
\lambda_{\max}(U^\top U).
}
$$

Khi đó:

$$
UU^\top\preceq\lambda I,
$$

nên:

$$
\Delta^\top UU^\top\Delta
\le
\lambda\|\Delta\|^2.
$$

Do đó:

$$
\mathcal J(q^{(t)}+\Delta)
\le
\mathcal J(q^{(t)})
+
\sum_i
\left[
s_i^{(t)}\Delta_i
+
\frac12(d_i+\lambda)\Delta_i^2
\right].
$$

Vế phải tách hoàn toàn theo từng weight và có thể tối ưu song song.

### 13.1. Giá trị $\lambda$

Lý thuyết:

$$
\lambda
=
\lambda_{\max}(U^\top U)
$$

là giá trị nhỏ nhất bảo đảm upper bound.

Thực thi FP32:

$$
\boxed{
\lambda
=
1.01\,
\lambda_{\max}(U^\top U).
}
$$

Không tạo $UU^\top$. Chỉ tạo Gram matrix:

$$
G_U=U^\top U\in\mathbb R^{r\times r}.
$$

PyTorch:

```python
def spectral_lambda(U, safety=1.01):
    gram = U.transpose(-1, -2) @ U
    eigvals = torch.linalg.eigvalsh(gram)
    return safety * eigvals[..., -1]
```

Tính một lần cho mỗi group rồi cache qua tất cả iterations.

---

## 14. Continuous assignment target

Upper bound theo weight $i$:

$$
s_i^{(t)}\Delta_i
+
\frac12(d_i+\lambda)\Delta_i^2.
$$

Nghiệm liên tục:

$$
\Delta_i^*
=
-\frac{s_i^{(t)}}{d_i+\lambda}.
$$

Target:

$$
\boxed{
t_i^{(t)}
=
q_i^{(t)}
-
\frac{s_i^{(t)}}{d_i+\lambda}.
}
$$

Sau đó projection vào scalar codebook:

$$
\boxed{
a_i^{(t+1)}
=
\arg\min_{k\in\{1,\ldots,K\}}
\left|c_k-t_i^{(t)}\right|.
}
$$

Vì mỗi $c_k$ là scalar, đây vẫn là scalar quantization.

Mọi $i$ được xử lý song song.

---

## 15. Tie-breaking

Để tránh labels dao động khi hai codewords cách target bằng nhau:

1. nếu current codeword là một minimizer, giữ current label;
2. nếu không, chọn index nhỏ nhất trong các minimizers.

Pseudocode:

```python
dist = (target[..., None] - codebook[..., None, :]).abs()
best_dist = dist.min(dim=-1).values

current_dist = dist.gather(
    dim=-1,
    index=labels[..., None],
).squeeze(-1)

keep_current = current_dist <= best_dist + tie_tol

new_labels = dist.argmin(dim=-1)
new_labels = torch.where(keep_current, labels, new_labels)
```

---

## 16. Chứng minh assignment loss không tăng

Gọi upper bound:

$$
M(q\mid q^{(t)})
=
\mathcal J(q^{(t)})
+
(s^{(t)})^\top(q-q^{(t)})
+
\frac12
(q-q^{(t)})^\top(D+\lambda I)(q-q^{(t)}).
$$

Vì:

$$
D+\lambda I\succeq D+UU^\top,
$$

ta có:

$$
M(q\mid q^{(t)})
\ge
\mathcal J(q)
$$

với mọi $q$, và:

$$
M(q^{(t)}\mid q^{(t)})
=
\mathcal J(q^{(t)}).
$$

Projection nearest-codeword tối ưu chính xác $M$ trên tập codebook product:

$$
q_i\in\mathcal C.
$$

Vì $q^{(t)}$ cũng là nghiệm khả thi:

$$
M(q^{(t+1)}\mid q^{(t)})
\le
M(q^{(t)}\mid q^{(t)}).
$$

Suy ra:

$$
\boxed{
\mathcal J(q^{(t+1)})
\le
M(q^{(t+1)}\mid q^{(t)})
\le
M(q^{(t)}\mid q^{(t)})
=
\mathcal J(q^{(t)}).
}
$$

Vậy assignment update là monotonic.

---

## 17. Assignment update pseudocode

```python
def parallel_mm_assignment(
    w,
    g,
    d,
    U,
    codebook,
    labels,
    beta,
    lambda_,
    tie_tol=0.0,
):
    q = codebook[labels]
    e = q - w

    # h = U^T e
    h = U.transpose(-1, -2) @ e

    # s = beta*g + D*e + U*h
    grad = beta * g + d * e + U @ h

    # Continuous MM target
    target = q - grad / (d + lambda_)

    # Exact nearest scalar codeword
    new_labels = nearest_codeword_with_current_tie_break(
        target=target,
        codebook=codebook,
        current_labels=labels,
        tie_tol=tie_tol,
    )

    return new_labels
```

---

# Phần V — Exact DLR Codebook Update

## 18. Bài toán codebook

Giữ labels $a$ cố định.

Cluster $k$:

$$
C_k=\{i:a_i=k\}.
$$

Ta giải:

$$
\boxed{
\min_{c_1,\ldots,c_K}
\mathcal J(a,c).
}
$$

Với labels cố định, đây là một convex quadratic problem theo $K$ scalar codewords.

Không cần:

- SGD;
- Lloyd iterations;
- coordinate descent;
- sorting weights.

---

## 19. Cluster statistics

Với mỗi active cluster $k$, tính:

$$
\boxed{
A_k=\sum_{i\in C_k}d_i
}
$$

$$
\boxed{
B_k=\sum_{i\in C_k}d_iw_i
}
$$

$$
\boxed{
G_k=\sum_{i\in C_k}g_i
}
$$

$$
\boxed{
m_k=\sum_{i\in C_k}U_{i,:}\in\mathbb R^r
}
$$

và:

$$
\boxed{
b_k=B_k-\beta G_k.
}
$$

Tính một vector constant theo group:

$$
\boxed{
z=U^\top w.
}
$$

---

## 20. Derivation của codeword tối ưu

Low-rank state dưới labels cố định:

$$
h
=
U^\top(q-w)
=
\sum_{k=1}^{K}m_kc_k-z.
$$

Đạo hàm loss theo $c_k$:

$$
\frac{\partial\mathcal J}{\partial c_k}
=
\beta G_k
+
A_kc_k
-
B_k
+
m_k^\top h.
$$

Tại optimum:

$$
\beta G_k+A_kc_k-B_k+m_k^\top h=0.
$$

Do đó:

$$
\boxed{
c_k
=
\frac{b_k-m_k^\top h}{A_k}.
}
$$

Nhưng $h$ phụ thuộc vào tất cả codewords. Thay công thức trên vào:

$$
h
=
\sum_km_k
\frac{b_k-m_k^\top h}{A_k}
-z.
$$

Suy ra:

$$
\left(
I_r+
\sum_k\frac{m_km_k^\top}{A_k}
\right)h
=
\sum_k\frac{m_kb_k}{A_k}
-z.
$$

Đặt:

$$
\boxed{
Q
=
\sum_k\frac{m_km_k^\top}{A_k}
}
$$

và:

$$
\boxed{
v
=
\sum_k\frac{m_kb_k}{A_k}
-z.
}
$$

Giải hệ:

$$
\boxed{
(I_r+Q)h=v.
}
$$

Sau đó:

$$
\boxed{
c_k
=
\frac{b_k-m_k^\top h}{A_k}.
}
$$

Chỉ cần giải một hệ $r\times r$. Với $r=4$, đây là hệ $4\times4$.

---

## 21. Empty clusters

Nếu cluster $k$ rỗng:

$$
A_k=0.
$$

Loss không phụ thuộc vào $c_k$, vì không weight nào đang dùng codeword đó.

Xử lý sạch và không thay đổi objective:

- giữ nguyên codeword cũ;
- loại cluster rỗng khỏi các tổng tạo $Q$ và $v$.

Không ép chuyển weight sang cluster rỗng, vì đó là một heuristic khác.

Cluster rỗng vẫn có thể được tái sử dụng ở assignment update sau nếu codeword cũ trở thành nearest target.

---

## 22. Codebook update pseudocode

```python
def exact_dlr_codebook_update(
    w,
    g,
    d,
    U,
    labels,
    old_codebook,
    beta,
    K,
):
    # Cluster reductions
    A = scatter_sum(d, labels, K)          # [K]
    B = scatter_sum(d * w, labels, K)      # [K]
    G = scatter_sum(g, labels, K)          # [K]
    M = scatter_sum_rows(U, labels, K)     # [K, r]

    b = B - beta * G
    z = U.transpose(-1, -2) @ w            # [r]

    active = A > 0

    A_a = A[active]
    b_a = b[active]
    M_a = M[active]

    # Q = sum_k m_k m_k^T / A_k
    Q = torch.einsum(
        "kr,ks,k->rs",
        M_a,
        M_a,
        1.0 / A_a,
    )

    # v = sum_k m_k b_k / A_k - z
    v = (M_a * (b_a / A_a)[:, None]).sum(dim=0) - z

    R = torch.eye(U.shape[-1], device=U.device, dtype=U.dtype) + Q
    h = torch.linalg.solve(R, v)

    new_codebook = old_codebook.clone()
    new_codebook[active] = (
        b_a - M_a @ h
    ) / A_a

    return new_codebook
```

---

## 23. Sort codebook sau update

Exact codebook update có thể làm codeword đổi thứ tự.

Để nearest-codeword search nhanh hơn, sort codebook:

```python
sorted_codebook, perm = torch.sort(codebook)
```

Cần remap labels để quantized weights $q$ không đổi.

Nếu:

```python
sorted_codebook[j] = old_codebook[perm[j]]
```

thì inverse permutation:

```python
inv_perm = torch.empty_like(perm)
inv_perm[perm] = torch.arange(K, device=perm.device)
new_labels = inv_perm[old_labels]
```

Việc sort và remap này giữ nguyên:

$$
q_i=c_{a_i}
$$

nên giữ nguyên objective chính xác.

---

# Phần VI — Thuật toán đầy đủ

## 24. Outer alternating algorithm

```python
def quantize_group_dlr(
    w,
    g,
    d,
    U,
    K,
    beta=0.5,
    max_outer_iters=8,
    rel_tol=1e-7,
    lambda_safety=1.01,
):
    # All solver arithmetic should be FP32
    w = w.float()
    g = g.float()
    d = d.float()
    U = U.float()

    # Constant caches
    z = U.transpose(-1, -2) @ w
    lambda_ = spectral_lambda(U, safety=lambda_safety)

    # ----- Initialization -----
    x = continuous_dlr_target(w, g, d, U, beta)
    labels = initialize_labels_from_target(x, d, U, K)

    codebook = initial_placeholder_codebook(x, labels, K)
    codebook = exact_dlr_codebook_update(
        w=w,
        g=g,
        d=d,
        U=U,
        labels=labels,
        old_codebook=codebook,
        beta=beta,
        K=K,
    )
    codebook, labels = sort_codebook_and_remap_labels(
        codebook,
        labels,
    )

    old_loss = dlr_loss(
        w, g, d, U, codebook, labels, beta
    )

    # ----- Alternating optimization -----
    for _ in range(max_outer_iters):
        old_labels = labels.clone()

        # A. Assignment update
        labels = parallel_mm_assignment(
            w=w,
            g=g,
            d=d,
            U=U,
            codebook=codebook,
            labels=labels,
            beta=beta,
            lambda_=lambda_,
        )

        # B. Exact codebook update
        codebook = exact_dlr_codebook_update(
            w=w,
            g=g,
            d=d,
            U=U,
            labels=labels,
            old_codebook=codebook,
            beta=beta,
            K=K,
        )

        codebook, labels = sort_codebook_and_remap_labels(
            codebook,
            labels,
        )

        new_loss = dlr_loss(
            w, g, d, U, codebook, labels, beta
        )

        # Numerical safety assertion
        loss_scale = torch.clamp(old_loss.abs(), min=1.0)
        if new_loss > old_loss + 1e-6 * loss_scale:
            raise RuntimeError("DLR loss unexpectedly increased")

        labels_unchanged = torch.equal(labels, old_labels)
        relative_drop = (old_loss - new_loss).abs() / loss_scale

        old_loss = new_loss

        if labels_unchanged or relative_drop <= rel_tol:
            break

    return codebook, labels
```

### Ghi chú về `initial_placeholder_codebook`

Codebook trước lần exact update đầu tiên chỉ cần có shape `[K]`.

Ví dụ lấy weighted mean của từng initial cluster hoặc median của $x$ trong cluster. Exact DLR update sẽ thay toàn bộ active codewords ngay sau đó, nên placeholder không ảnh hưởng kết quả với active clusters.

Nó chỉ được giữ lại cho empty clusters; initialization đã ép nonempty nên trường hợp đó không xảy ra ở vòng đầu.

---

# Phần VII — Tính hội tụ

## 25. Assignment step

Với:

$$
\lambda\ge\lambda_{\max}(U^\top U),
$$

parallel MM assignment bảo đảm:

$$
\mathcal J(a^{(t+1)},c^{(t)})
\le
\mathcal J(a^{(t)},c^{(t)}).
$$

---

## 26. Codebook step

Exact codebook update giải:

$$
c^{(t+1)}
=
\arg\min_c
\mathcal J(a^{(t+1)},c).
$$

Do đó:

$$
\mathcal J(a^{(t+1)},c^{(t+1)})
\le
\mathcal J(a^{(t+1)},c^{(t)}).
$$

---

## 27. Một outer iteration

Ghép hai bước:

$$
\boxed{
\mathcal J(a^{(t+1)},c^{(t+1)})
\le
\mathcal J(a^{(t)},c^{(t)}).
}
$$

Objective bị chặn dưới vì quadratic curvature là PSD. Do đó dãy loss hội tụ.

Với các điều kiện:

- tie-breaking deterministic;
- chỉ đổi label khi majorizer thực sự tốt hơn hoặc current label không phải minimizer;
- codebook update có nghiệm duy nhất trên active clusters;
- empty cluster được xử lý deterministic;

thuật toán hội tụ tới một fixed point của alternating MM procedure.

Không có guarantee tìm global optimum của bài toán discrete non-convex.

---

# Phần VIII — GPU Implementation Không Đổi Thuật Toán

## 28. Nguyên tắc chung

Các tối ưu dưới đây chỉ thay đổi cách thực thi, không thay đổi objective hoặc update rule:

1. batch nhiều groups/rows;
2. không materialize $UU^\top$;
3. cache mọi đại lượng không đổi;
4. FP32 cho reductions, eigensolver và linear solve;
5. fuse elementwise kernels;
6. dùng exact nearest-codeword search;
7. dùng exact segmented reductions cho codebook statistics.

---

## 29. Batch nhiều groups

Giả sử:

```text
B = number of groups processed together
n = weights per group
r = low rank
K = codebook size
```

Shapes:

```text
w          [B, n]
g          [B, n]
d          [B, n]
U          [B, n, r]
labels     [B, n]
codebook   [B, K]
lambda     [B]
```

Các phép chính đều batched:

```python
q = torch.gather(codebook, 1, labels)
e = q - w

h = torch.einsum("bnr,bn->br", U, e)
grad = beta * g + d * e + torch.einsum("bnr,br->bn", U, h)

target = q - grad / (d + lambda_[:, None])
```

---

## 30. Cache một lần

Các đại lượng không đổi qua iterations:

```text
z = U^T w
row_norm_U2 = ||U_i||^2
rho = d + row_norm_U2
lambda = 1.01 * lambda_max(U^T U)
```

Initialization caches:

```text
inv_d
D^{-1} U
I + U^T D^{-1} U
```

Không tính lại nếu $w,g,d,U$ không đổi.

---

## 31. Không materialize $UU^\top$

Không bao giờ tạo tensor `[B, n, n]`.

Thay:

```python
H_lowrank_e = (U @ U.T) @ e
```

bằng:

```python
h = U.transpose(-1, -2) @ e
H_lowrank_e = U @ h
```

Complexity:

$$
O(nr)
$$

thay vì:

$$
O(n^2).
$$

---

## 32. Exact nearest-codeword trên GPU

### 32.1. Exhaustive search

Với $K=8$ hoặc $K=16$, exhaustive search thường nhanh:

```python
dist = (target[..., None] - codebook[:, None, :]).abs()
new_labels = dist.argmin(dim=-1)
```

Nhưng tensor `[B,n,K]` có thể tốn memory.

### 32.2. Sorted search

Vì codebook scalar và đã sort:

1. dùng `searchsorted`;
2. lấy codeword trái và phải;
3. so sánh đúng hai khoảng cách.

Đây là **exact**, không phải approximation.

Complexity:

$$
O(n\log K).
$$

Một Triton kernel có thể fuse:

```text
searchsorted
+ compare left/right
+ current-label tie-break
+ write labels
```

Không cần tạo `[B,n,K]`.

### 32.3. Tie behavior phải giống nhau

Exhaustive và sorted search chỉ cho cùng kết quả nếu dùng cùng rule:

- giữ current label khi current codeword cùng đạt min distance;
- nếu không, chọn index nhỏ hơn.

---

## 33. Fused assignment kernel

Có thể fuse:

```text
q = codebook[label]
e = q - w
grad = beta*g + d*e + dot(U_i, h)
target = q - grad/(d+lambda)
nearest-codeword
write new label
```

$h=U^\top e$ vẫn cần một reduction riêng trước kernel.

Hai-pass GPU implementation:

```text
Pass 1: compute e and h = U^T e
Pass 2: fused grad + target + nearest-codeword
```

Không thay đổi thuật toán.

---

## 34. Fast exact codebook reductions

Cần tính cho mỗi `[group, cluster]`:

```text
A = sum d
B = sum d*w
G = sum g
M = sum U
```

Có hai lựa chọn.

### 34.1. `scatter_add_`

Dễ triển khai và nhanh:

```python
A.scatter_add_(1, labels, d)
B.scatter_add_(1, labels, d * w)
G.scatter_add_(1, labels, g)
```

Với $M$, expand labels theo rank dimension.

### 34.2. Segmented reduction

Để reproducibility tốt hơn:

1. sort indices theo composite key `group_id * K + label`;
2. segmented reduce;
3. reshape thành `[B,K,...]`.

Về toán học, hai cách cho cùng các sums. Sai khác nếu có chỉ do thứ tự cộng floating-point.

Nếu cần bitwise determinism, dùng segmented reduction FP32.

---

## 35. Batched small linear solves

Codebook update cần giải:

$$
(I_r+Q_b)h_b=v_b
$$

cho từng group $b$.

Shapes:

```text
R [B, r, r]
v [B, r]
```

Dùng:

```python
h = torch.linalg.solve(R, v.unsqueeze(-1)).squeeze(-1)
```

Không dùng explicit inverse:

```python
# Avoid
h = torch.linalg.inv(R) @ v
```

`solve` nhanh và ổn định hơn.

---

## 36. Precision

Khuyến nghị:

```text
weights/model storage: FP16 hoặc BF16
solver arithmetic: FP32
```

Bắt buộc FP32 cho:

- `U.T @ U`;
- `eigvalsh`;
- Woodbury solve;
- cluster reductions;
- exact DLR loss;
- codebook linear solve.

Codebook có thể cast về model dtype sau khi optimization kết thúc.

Dùng FP16 trong reductions có thể thay đổi labels ở các target sát boundary, tức có thể thay đổi kết quả thuật toán.

---

## 37. Stopping nhanh mà không đổi fixed point

Các điều kiện dừng an toàn:

1. labels không đổi;
2. exact loss improvement nhỏ hơn tolerance;
3. max iterations là safety cap.

Nếu muốn kết quả không phụ thuộc cap, điều kiện chính nên là:

```python
if torch.equal(new_labels, old_labels):
    stop
```

Vì codebook update là deterministic theo labels, labels không đổi sau một outer iteration thường cho fixed point.

Khuyến nghị:

```text
max_outer_iters = 8
rel_tol = 1e-7
```

---

## 38. Numerical guard

Theo lý thuyết loss không tăng. Trong implementation nên kiểm tra:

```python
new_loss <= old_loss + numerical_tolerance
```

Nếu vi phạm:

1. recompute step hoàn toàn ở FP32;
2. tăng `lambda_safety`, ví dụ từ `1.01` lên `1.05`;
3. không nhận state mới nếu vẫn tăng.

Đây là numerical fallback, không phải thành phần chính của thuật toán.

---

# Phần IX — Complexity và Memory

## 39. Initialization

Continuous target bằng Woodbury:

$$
O(nr^2+r^3).
$$

Sort:

$$
O(n\log n).
$$

Initial partition:

$$
O(n).
$$

Initial codebook update:

$$
O(nr+Kr^2+r^3).
$$

---

## 40. Mỗi outer iteration

Assignment:

$$
O(nr+n\log K)
$$

với sorted nearest-codeword search, hoặc:

$$
O(nr+nK)
$$

với exhaustive search.

Codebook update:

$$
O(nr+Kr^2+r^3).
$$

Với fixed small $r$ và $K$, mỗi outer iteration gần tuyến tính theo $n$.

---

## 41. Memory

Per group:

```text
w,g,d,q,e,target,labels: O(n)
U: O(nr)
codebook and stats: O(Kr)
small matrices: O(r^2)
```

Không có tensor $O(n^2)$.

---

# Phần X — Recommended Defaults

```yaml
beta: 0.5
rank: 4
lambda_safety: 1.01
max_outer_iters: 8
relative_loss_tolerance: 1.0e-7
solver_dtype: float32
codebook_size:
  3-bit: 8
  4-bit: 16
assignment:
  mode: parallel_mm_all_codewords
  neighborhood_restriction: false
codebook_update:
  mode: exact_dlr
empty_cluster:
  mode: keep_previous_codeword
tie_break:
  prefer_current_label: true
```

---

# Phần XI — Unit Tests Cho Codex

## 42. Loss equivalence

So sánh:

$$
\frac12e^\top(D+UU^\top)e
$$

với:

$$
\frac12\sum_i d_ie_i^2+\frac12\|U^\top e\|^2.
$$

Sai số phải gần FP32 tolerance.

---

## 43. Spectral upper bound

Sinh random $\Delta$, kiểm tra:

$$
\Delta^\top UU^\top\Delta
\le
\lambda\|\Delta\|^2.
$$

Với:

$$
\lambda=1.01\lambda_{\max}(U^\top U).
$$

---

## 44. Assignment monotonicity

Với codebook cố định:

```text
loss_after_assignment <= loss_before_assignment
```

trên nhiều random seeds.

---

## 45. Codebook optimality

Sau exact codebook update, gradient theo active codewords phải gần 0:

$$
\beta G_k+A_kc_k-B_k+m_k^\top h\approx0.
$$

---

## 46. Codebook update monotonicity

Với labels cố định:

```text
loss_after_codebook <= loss_before_codebook
```

---

## 47. Sort/remap invariance

Trước và sau sort codebook + remap labels:

```text
q_before == q_after
loss_before == loss_after
```

trong numerical tolerance.

---

## 48. Batched/unbatched equivalence

Chạy cùng groups:

- từng group riêng;
- cả batch.

Kết quả labels và codebooks phải giống nhau ngoài sai khác roundoff được quy định.

---

## 49. Empty-cluster handling

Tạo labels có cluster rỗng. Kiểm tra:

- không chia cho 0;
- active codewords đúng;
- empty codeword giữ nguyên;
- loss không tăng.

---

# Phần XII — Tóm tắt Cho Implementation

## 50. Công thức chính

### Objective

$$
\boxed{
\mathcal J(q)
=
\beta g^\top(q-w)
+
\frac12\sum_i d_i(q_i-w_i)^2
+
\frac12\|U^\top(q-w)\|^2
}
$$

### Initialization target

$$
\boxed{
x=w-\beta(D+UU^\top)^{-1}g
}
$$

tính bằng Woodbury.

### Spectral parameter

$$
\boxed{
\lambda=1.01\,\lambda_{\max}(U^\top U)
}
$$

### Assignment gradient

$$
\boxed{
s=\beta g+D(q-w)+UU^\top(q-w)
}
$$

tính dưới dạng:

$$
h=U^\top(q-w),
\qquad
s=\beta g+D(q-w)+Uh.
$$

### Assignment target

$$
\boxed{
t_i=q_i-\frac{s_i}{d_i+\lambda}
}
$$

### Assignment

$$
\boxed{
a_i^{\mathrm{new}}
=
\arg\min_k|c_k-t_i|
}
$$

### Codebook system

$$
A_k=\sum_{a_i=k}d_i,
\quad
b_k=\sum_{a_i=k}(d_iw_i-\beta g_i),
\quad
m_k=\sum_{a_i=k}U_{i,:}.
$$

$$
Q=\sum_k\frac{m_km_k^\top}{A_k},
$$

$$
v=\sum_k\frac{m_kb_k}{A_k}-U^\top w,
$$

$$
\boxed{
(I+Q)h=v
}
$$

$$
\boxed{
c_k=\frac{b_k-m_k^\top h}{A_k}.
}
$$

---

## 51. Final algorithm

```text
Input: w, g, d, U, K, beta

Precompute:
    lambda = 1.01 * largest_eigenvalue(U^T U)

Initialize:
    x = w - beta * (D + U U^T)^(-1) g   # Woodbury
    labels = curvature-balanced quantile partition of sorted x
    codebook = exact DLR codebook update(labels)
    sort codebook and remap labels

Repeat:
    q = codebook[labels]
    e = q - w
    h = U^T e
    grad = beta*g + d*e + U*h

    target = q - grad / (d + lambda)
    labels = nearest scalar codeword(target)
             with current-label tie-breaking

    codebook = exact DLR codebook update(labels)
    sort codebook and remap labels

Until:
    labels unchanged
    or exact loss improvement below tolerance

Output:
    scalar codebook and scalar label for every weight
```

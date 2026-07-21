# Method A: Constrained End-Loss Scalar Quantization
## Direct Streaming \(D+UU^\top\) Curvature + Parallel MM Assignment + Exact Codebook Update

> **Mục tiêu của tài liệu:** đặc tả đầy đủ một phương pháp **weight-only non-uniform scalar quantization** có thể triển khai bằng PyTorch/Triton trên GPU.
>
> Mỗi scalar weight được thay bằng một trong \(K\) scalar codewords. Đây **không phải vector quantization**.
>
> Phiên bản này giữ nguyên formulation và solver của Method A, nhưng thay dense GuidedQuant-style Hessian bằng một biểu diễn:
>
> \[
> \boxed{
> H \approx D+UU^\top
> }
> \]
>
> được collect **trực tiếp từ calibration statistics**, không bao giờ dựng dense \(H\) rồi mới phân rã.

---

# 1. Tóm tắt thay đổi so với phiên bản dense Hessian

Phiên bản cũ dùng curvature grouped dạng:

\[
H_k
=
\frac1T
X^\top
\operatorname{Diag}(s_k)
X
\in
\mathbb R^{d_{\mathrm{in}}\times d_{\mathrm{in}}}.
\]

Vấn đề lớn nhất là chi phí:

\[
O(Td_{\mathrm{in}}^2)
\]

cho mỗi curvature group, mỗi loại loss và mỗi module.

Phiên bản mới giữ nguyên statistical target:

\[
H_k
=
\frac1T
X^\top
\operatorname{Diag}(s_k)
X,
\]

nhưng chỉ collect một approximation:

\[
\boxed{
H_k
\approx
D_k+U_kU_k^\top
}
\]

với:

- \(D_k\) là diagonal không âm;
- \(U_k\in\mathbb R^{d_{\mathrm{in}}\times r}\);
- \(r\ll d_{\mathrm{in}}\).

Quan trọng:

\[
\boxed{
\text{Không tính dense }H\text{ trước.}
}
\]

\(D\) và \(U\) được xây trực tiếp từ streaming activation/gradient statistics.

---

# 2. Reference và phạm vi reuse

Reference GuidedQuant:

- GitHub: https://github.com/snu-mllab/GuidedQuant
- Paper: https://openreview.net/forum?id=ZawsPjlIGu

Ta chỉ reuse các ý sau:

1. curvature ở weight-row space được xây từ layer input \(x_t\) và layer-output gradient \(\delta_{t,j}\);
2. nhiều output rows có thể share một curvature group;
3. NLL và KL phải dùng đúng upstream loss/source khác nhau.

Ta **không** reuse dense Hessian accumulation.

Ta **không** chạy dense Hessian rồi dùng Cholesky, eigendecomposition hoặc SVD để phân rã.

Ta **không** dùng LNQ coordinate descent làm assignment.

---

# 3. Scope và notation

Xét một linear layer PyTorch:

\[
Y=XW^\top,
\]

với:

\[
W\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}}.
\]

Mỗi output row:

\[
w_j\in\mathbb R^{d_{\mathrm{in}}}.
\]

Calibration activations:

\[
X\in\mathbb R^{T\times d_{\mathrm{in}}}.
\]

Trong tài liệu:

- \(T\): tổng valid calibration tokens;
- \(n=d_{\mathrm{in}}\);
- \(K=2^b\): số codewords;
- \(r\): low-rank dimension;
- quantization group: một weight row;
- curvature group: một nhóm output rows cùng dùng chung \(D,U\).

Một row sau quantization:

\[
q\in\mathbb R^n.
\]

Error:

\[
\boxed{
e=q-w.
}
\]

Codebook:

\[
\mathcal C=\{c_1,\ldots,c_K\}.
\]

---

# 4. Bài toán gốc

Ta tối ưu end NLL nhưng giới hạn thay đổi output distribution và độ xa trong weight space:

\[
\boxed{
\min_{q\in\mathcal Q_K}
L_{\mathrm{NLL}}(q)
}
\]

subject to:

\[
\boxed{
D_{\mathrm{KL}}
\left(
p_w(\cdot|x)
\|
p_q(\cdot|x)
\right)
\le
\epsilon_{\mathrm{KL}}
}
\]

và:

\[
\boxed{
\frac12\|q-w\|_2^2
\le
\epsilon_w.
}
\]

---

# 5. Taylor expansion của NLL

Đặt:

\[
e=q-w.
\]

Khai triển quanh full-precision weight:

\[
L_{\mathrm{NLL}}(w+e)
\approx
L_{\mathrm{NLL}}(w)
+
g^\top e
+
\frac12e^\top H_{\mathrm{NLL}}e.
\]

Trong đó:

\[
\boxed{
g=\nabla_wL_{\mathrm{NLL}}(w)
}
\]

là signed gradient theo weight.

Local objective:

\[
\boxed{
g^\top e
+
\frac12e^\top H_{\mathrm{NLL}}e.
}
\]

Trong implementation:

\[
\boxed{
H_{\mathrm{NLL}}
\approx
D_N+U_NU_N^\top.
}
\]

Do đó:

\[
\boxed{
g^\top e
+
\frac12
\left[
\sum_i d_{N,i}e_i^2
+
\|U_N^\top e\|_2^2
\right].
}
\]

---

# 6. Taylor expansion của KL

Tại teacher = student:

\[
D_{\mathrm{KL}}(p_w\|p_w)=0,
\]

và gradient bậc nhất theo perturbation bằng 0.

Do đó:

\[
D_{\mathrm{KL}}(p_w\|p_{w+e})
\approx
\frac12e^\top H_{\mathrm{KL}}e.
\]

Trong implementation:

\[
\boxed{
H_{\mathrm{KL}}
\approx
D_K+U_KU_K^\top.
}
\]

KL local constraint:

\[
\boxed{
\frac12
\left[
\sum_i d_{K,i}e_i^2
+
\|U_K^\top e\|_2^2
\right]
\le
\epsilon_{\mathrm{KL}}.
}
\]

---

# 7. Phải tách riêng \(g\), \(H_{\mathrm{NLL}}\), \(H_{\mathrm{KL}}\)

## 7.1. Signed gradient \(g\)

Nguồn duy nhất:

\[
\boxed{
\text{ground-truth next-token NLL}
}
\]

Với:

\[
L_{\mathrm{NLL}}
=
-\frac1T
\sum_t
\log p_w(y_t^{\mathrm{true}}|x_{\le t}).
\]

Ta lấy:

\[
\boxed{
g=\nabla_wL_{\mathrm{NLL}}.
}
\]

Quy tắc:

- không square \(g\);
- không lấy absolute;
- không lấy từ KL;
- không thay bằng Fisher diagonal;
- không lấy \(g/H\) làm initialization.

---

## 7.2. \(H_{\mathrm{NLL}}\)

Nguồn upstream:

\[
\boxed{
\text{ground-truth NLL end-loss gradients}
}
\]

Tại linear layer:

\[
\delta_{t,j}^{N}
=
\frac{\partial \ell_t^{\mathrm{NLL}}}
{\partial Z_{t,j}}.
\]

Curvature sample:

\[
\delta_{t,j}^{N\,2}x_tx_t^\top.
\]

Sau output-channel grouping:

\[
s_{t,k}^{N}
=
\frac1{|J_k|}
\sum_{j\in J_k}
\left(\delta_{t,j}^{N}\right)^2.
\]

Target curvature:

\[
H_{N,k}
=
\frac1T
X^\top
\operatorname{Diag}(s_k^N)
X.
\]

Ta chỉ approximate target này bằng:

\[
\boxed{
H_{N,k}
\approx
D_{N,k}+U_{N,k}U_{N,k}^\top.
}
\]

---

## 7.3. \(H_{\mathrm{KL}}\)

Nguồn upstream:

\[
\boxed{
\text{teacher output distribution}
}
\]

Không dùng ground-truth labels.

Với mỗi valid token:

\[
y_t^{(m)}
\sim
p_w(\cdot|x_{\le t}).
\]

Score:

\[
a_t^{(m)}
=
\log p_w(y_t^{(m)}|x_{\le t}).
\]

Layer-output score gradient:

\[
\delta_{t,j}^{K,(m)}
=
\frac{\partial a_t^{(m)}}
{\partial Z_{t,j}}.
\]

Sensitivity:

\[
s_{t,k}^{K}
=
\frac1{|J_k|M}
\sum_{j\in J_k}
\sum_{m=1}^{M}
\left(
\delta_{t,j}^{K,(m)}
\right)^2.
\]

Target curvature:

\[
H_{K,k}
=
\frac1T
X^\top
\operatorname{Diag}(s_k^K)
X.
\]

Approximation:

\[
\boxed{
H_{K,k}
\approx
D_{K,k}+U_{K,k}U_{K,k}^\top.
}
\]

---

# 8. Không được trộn NLL và KL collector

Hai curvature có cùng structural formula:

\[
X^\top\operatorname{Diag}(s)X,
\]

nhưng sensitivity \(s\) đến từ hai nguồn khác nhau.

NLL:

\[
s^N
\leftarrow
\text{ground-truth NLL gradients}.
\]

KL:

\[
s^K
\leftarrow
\text{teacher-score Fisher gradients}.
\]

Không được:

- dùng ground-truth labels cho KL;
- dùng teacher score để thay signed \(g\);
- dùng chung một \(D,U\) rồi gọi là cả \(H_N\) và \(H_K\);
- average hai nguồn gradient trước khi square.

---

# 9. Local constrained problem với \(D+UU^\top\)

Bài toán cuối cùng:

\[
\boxed{
\min_{q\in\mathcal Q_K}
g^\top e
+
\frac12
\left[
e^\top D_Ne
+
\|U_N^\top e\|^2
\right]
}
\]

subject to:

\[
\boxed{
\frac12
\left[
e^\top D_Ke
+
\|U_K^\top e\|^2
\right]
\le
\epsilon_{\mathrm{KL}}
}
\]

và:

\[
\boxed{
\frac12\|e\|^2
\le
\epsilon_w.
}
\]

---

# 10. Initialization

Dùng scalar quantization reference:

\[
\boxed{
q_0=Q_{\mathrm{sqllm}}(w).
}
\]

Đặt:

\[
e_0=q_0-w.
\]

Budgets per row:

\[
\boxed{
\epsilon_{\mathrm{KL}}
=
\frac12
\left[
e_0^\top D_Ke_0
+
\|U_K^\top e_0\|^2
\right].
}
\]

\[
\boxed{
\epsilon_w
=
\frac12\|e_0\|^2.
}
\]

Theo construction, \(q_0\) luôn feasible.

Không dùng:

\[
w-H^{-1}g
\]

hoặc:

\[
w-\frac{g}{d}
\]

làm initialization.

---

# 11. Lagrangian

Dual variables:

\[
\mu\ge0,
\qquad
\nu\ge0.
\]

Fixed-dual objective:

\[
J(e)
=
g^\top e
+
\frac12e^\top Ae.
\]

Với:

\[
A
=
H_N+\mu H_K+\nu I.
\]

Thay \(D+UU^\top\):

\[
\boxed{
A
=
D_A
+
U_NU_N^\top
+
\mu U_KU_K^\top,
}
\]

trong đó:

\[
\boxed{
D_A
=
D_N+\mu D_K+\nu I.
}
\]

Vì \(D_N,D_K\ge0\):

\[
H_N\succeq0,
\qquad
H_K\succeq0.
\]

---

# 12. Cách collect trực tiếp \(D+UU^\top\)

## 12.1. Weighted sample matrix

Với một curvature group:

\[
H
=
\frac1T
X^\top
\operatorname{Diag}(s)X.
\]

Đặt:

\[
\boxed{
Z
=
\frac1{\sqrt T}
\operatorname{Diag}(\sqrt{s})X.
}
\]

Khi đó:

\[
\boxed{
H=Z^\top Z.
}
\]

Mỗi token đóng góp weighted activation:

\[
z_t
=
\sqrt{\frac{s_t}{T}}x_t.
\]

Ta không lưu toàn bộ \(Z\).

---

## 12.2. Random sketch

Chọn fixed sketch matrix:

\[
\Omega
\in
\mathbb R^{n\times r}.
\]

\(\Omega\) được tạo deterministically từ seed cố định của module/group.

Recommended:

- Rademacher entries \(\pm1/\sqrt r\), hoặc
- Gaussian normalized.

Không đổi \(\Omega\) giữa các calibration batches của cùng group.

---

## 12.3. Streaming statistics

Với batch:

\[
X_b\in\mathbb R^{T_b\times n},
\]

và sensitivity:

\[
s_b\in\mathbb R^{T_b},
\]

đặt:

\[
Z_b
=
\operatorname{Diag}(\sqrt{s_b})X_b.
\]

Nếu normalization chia cuối collector, không cần đưa \(1/\sqrt T\) vào \(Z_b\).

Projection:

\[
P_b=Z_b\Omega.
\]

Accumulate:

\[
\boxed{
C
\mathrel{+}=
Z_b^\top P_b
}
\]

\[
\boxed{
B
\mathrel{+}=
P_b^\top P_b
}
\]

\[
\boxed{
h_{\mathrm{diag}}
\mathrel{+}=
\sum_{t\in b}z_t\odot z_t.
}
\]

Cuối cùng:

\[
C\leftarrow\frac CT,
\qquad
B\leftarrow\frac BT,
\qquad
h_{\mathrm{diag}}\leftarrow\frac{h_{\mathrm{diag}}}{T}.
\]

Shape:

\[
C\in\mathbb R^{n\times r},
\]

\[
B\in\mathbb R^{r\times r},
\]

\[
h_{\mathrm{diag}}\in\mathbb R^n.
\]

---

# 13. Từ sketch statistics sang \(U\)

Nyström approximation:

\[
H_{\mathrm{LR}}
=
CB^\dagger C^\top.
\]

Eigendecomposition ma trận nhỏ:

\[
B=V\Lambda V^\top.
\]

Chỉ giữ numerical positive eigenvalues:

\[
\lambda_i>\tau.
\]

Ngưỡng tự động:

\[
\boxed{
\tau
=
\epsilon_{\mathrm{mach}}
\cdot r
\cdot
\max_i|\lambda_i|.
}
\]

Không phải tunable damping.

Đặt:

\[
\boxed{
U
=
CV_{\mathrm{act}}
\Lambda_{\mathrm{act}}^{-1/2}.
}
\]

Khi đó:

\[
\boxed{
UU^\top
=
CB^\dagger C^\top.
}
\]

---

# 14. Cách tính \(D\)

Exact target diagonal:

\[
\operatorname{diag}(H)
=
h_{\mathrm{diag}}.
\]

Low-rank diagonal:

\[
\operatorname{diag}(UU^\top)_i
=
\sum_aU_{ia}^2.
\]

Residual diagonal:

\[
\boxed{
d_i
=
\max
\left(
h_{\mathrm{diag},i}
-
\sum_aU_{ia}^2,
0
\right).
}
\]

Vậy:

\[
\boxed{
H
\approx
D+UU^\top,
\qquad
D=\operatorname{Diag}(d).
}
\]

Clamp về zero chỉ sửa sai số floating point và bảo đảm PSD.

Không thêm damping thủ công vào \(D\) ở collector.

---

# 15. Tại sao approximation này tốt hơn dạng tách kỳ vọng cũ

Không dùng:

\[
E[\delta^2]E[xx^\top].
\]

Vì:

\[
E[\delta^2xx^\top]
\neq
E[\delta^2]E[xx^\top]
\]

nói chung.

Collector mới sketch trực tiếp weighted samples:

\[
z_t=\sqrt{s_t}x_t.
\]

Do đó vẫn giữ tương quan giữa:

\[
s_t
\quad\text{và}\quad
x_tx_t^\top.
\]

Đây là điểm quan trọng để approximation gần GuidedQuant target hơn.

---

# 16. Normalization bắt buộc phải đúng

Nếu loss được backward dưới dạng mean:

\[
L=\frac1T\sum_t\ell_t,
\]

thì autograd gradient đã chứa \(1/T\).

Nếu square trực tiếp, curvature bị thêm factor \(1/T^2\).

Recommended:

1. backward `loss_sum`;
2. collect per-token \(\delta_t\);
3. square sensitivity;
4. accumulate raw sums;
5. cuối cùng chia tổng valid tokens \(T\).

Signed weight gradient cuối cùng:

\[
g
=
\frac1T
\nabla_w
\sum_t\ell_t.
\]

Collector phải có unit test so sánh sum-loss và mean-loss đã undo scaling.

---

# 17. NLL statistics pass

Trong một NLL backward có thể đồng thời collect:

1. signed weight gradient \(g\);
2. layer inputs \(X\);
3. grouped NLL sensitivity \(s^N\);
4. sketch statistics \(C_N,B_N,h_{\mathrm{diag},N}\).

Pseudocode logic:

```text
forward model with ground-truth labels
compute summed NLL over valid tokens
backward once

for each hooked linear module:
    obtain X
    obtain d(loss_sum)/dZ
    group square gradients over output channels
    update NLL sketch statistics
    accumulate signed weight gradient
```

Không cần một backward riêng cho \(g\) và một backward riêng cho \(H_N\).

---

# 18. KL Fisher statistics pass

Teacher/model gốc sinh logits:

\[
p_t=\operatorname{softmax}(z_t).
\]

Sample pseudo-label:

\[
y_t^{(m)}\sim p_t.
\]

Score sum:

\[
A^{(m)}
=
\sum_t
\log p_t(y_t^{(m)}).
\]

Backward score để lấy:

\[
\delta^{K,(m)}.
\]

Sau đó group square gradients và update KL sketch.

Prototype đầu tiên:

\[
\boxed{
M=1
}
\]

để giảm compute.

Tăng \(M\) chỉ để giảm Monte-Carlo variance.

Không trộn các probe bằng cách average gradient trước rồi square. Phải:

\[
\frac1M
\sum_m
\delta_m^2.
\]

---

# 19. Streaming collector pseudocode

```python
def update_sketch(
    X,                 # [tokens, n]
    sensitivity,       # [tokens]
    omega,             # [n, r]
    C,                 # [n, r]
    B,                 # [r, r]
    diag_h,            # [n]
):
    X32 = X.float()
    s32 = sensitivity.float().clamp_min(0)

    Z = X32 * torch.sqrt(s32).unsqueeze(1)
    P = Z @ omega

    C.add_(Z.T @ P)
    B.add_(P.T @ P)
    diag_h.add_((Z * Z).sum(dim=0))
```

Cuối collector:

```python
C /= total_valid_tokens
B /= total_valid_tokens
diag_h /= total_valid_tokens

evals, evecs = torch.linalg.eigh(B)

eps = torch.finfo(B.dtype).eps
threshold = eps * B.shape[0] * evals.abs().max()

active = evals > threshold

U = (
    C
    @ evecs[:, active]
    @ torch.diag(evals[active].rsqrt())
)

D = (
    diag_h
    - U.square().sum(dim=1)
).clamp_min(0)
```

---

# 20. Curvature-vector product

Không dựng lại \(H\).

Với vector \(e\):

\[
\boxed{
He
=
d\odot e
+
U(U^\top e).
}
\]

Với row-vector convention:

\[
\boxed{
eH
=
e\odot d
+
(eU)U^\top.
}
\]

Batch rows:

```python
EH = E * d[None, :] + (E @ U) @ U.T
```

Chi phí:

\[
O(Bnr)
\]

thay vì:

\[
O(Bn^2).
\]

---

# 21. Quadratic form

Với:

\[
H=D+UU^\top,
\]

ta có:

\[
\boxed{
e^\top He
=
\sum_i d_ie_i^2
+
\|U^\top e\|_2^2.
}
\]

Dùng công thức này cho:

- NLL surrogate;
- KL constraint;
- fixed-dual objective;
- initialization budgets;
- debugging metrics.

---

# 22. Parallel MM assignment

Fixed-dual gradient theo \(q\):

\[
s
=
g+Ae.
\]

Với \(D+UU^\top\):

\[
\boxed{
s
=
g
+
d_N\odot e
+
U_N(U_N^\top e)
+
\mu
\left[
d_K\odot e
+
U_K(U_K^\top e)
\right]
+
\nu e.
}
\]

Batch implementation:

```python
EN = E * d_nll[None, :] + (E @ U_nll) @ U_nll.T
EK = E * d_kl[None, :]  + (E @ U_kl)  @ U_kl.T

S = (
    G
    + EN
    + mu[:, None] * EK
    + nu[:, None] * E
)
```

---

# 23. Safe diagonal majorizer

Ta cần:

\[
M=\operatorname{Diag}(m)\succeq A.
\]

Với:

\[
H=D+UU^\top,
\]

upper bound absolute row sum:

\[
\sum_j|H_{ij}|
\le
d_i
+
\sum_a
|U_{ia}|
\sum_j|U_{ja}|.
\]

Đặt:

\[
\boxed{
r_i(D,U)
=
d_i
+
\sum_a
|U_{ia}|c_a,
}
\]

với:

\[
c_a=\sum_j|U_{ja}|.
\]

Precompute:

\[
r_N=r(D_N,U_N),
\]

\[
r_K=r(D_K,U_K).
\]

Fixed-dual majorizer:

\[
\boxed{
m_i
=
r_{N,i}
+
\mu r_{K,i}
+
\nu.
}
\]

Code:

```python
def lowrank_row_majorizer(d, U):
    col_l1 = U.abs().sum(dim=0)
    return d + U.abs() @ col_l1
```

Đây là safe majorizer, dù có thể lỏng hơn exact dense absolute row sum.

---

# 24. MM target và assignment

Continuous target:

\[
\boxed{
t_i
=
q_i-\frac{s_i}{m_i}.
}
\]

Discrete update:

\[
\boxed{
q_i^+
=
\operatorname{NearestCodeword}(t_i).
}
\]

Tất cả coordinates update đồng thời.

Tie-breaking:

1. giữ current codeword nếu nằm trong tie;
2. nếu không, dùng deterministic index rule.

Nếu \(m_i=0\):

- \(s_i>0\): chọn codeword nhỏ nhất;
- \(s_i<0\): chọn codeword lớn nhất;
- \(s_i=0\): giữ nguyên.

Không thêm damping method-level chỉ để tránh chia zero.

---

# 25. Fixed-dual monotonicity

Vì:

\[
M\succeq A,
\]

surrogate MM majorizes objective.

Do đó với fixed:

\[
\mu,\nu,\mathcal C,
\]

Parallel MM assignment bảo đảm:

\[
\boxed{
J(q^{t+1})
\le
J(q^t).
}
\]

Guarantee áp dụng cho approximate curvature:

\[
\widehat H_N=D_N+U_NU_N^\top,
\]

\[
\widehat H_K=D_K+U_KU_K^\top.
\]

---

# 26. Exact codebook update

Representation:

\[
q=Pc.
\]

Fixed assignments \(P\).

Codebook system vẫn là:

\[
\boxed{
(P^\top AP)c
=
P^\top Aw-P^\top g.
}
\]

Đây vẫn là exact minimizer của fixed-label subproblem dưới approximate \(A\).

---

# 27. Tính \(P^\top AP\) không cần dense \(A\)

Đặt:

\[
d_A=d_N+\mu d_K+\nu.
\]

Ta có:

\[
A
=
D_A
+
U_NU_N^\top
+
\mu U_KU_K^\top.
\]

Vế trái:

\[
P^\top AP
=
P^\top D_AP
+
(P^\top U_N)(P^\top U_N)^\top
+
\mu(P^\top U_K)(P^\top U_K)^\top.
\]

Vì \(P\) one-hot theo coordinate:

\[
P^\top D_AP
\]

là diagonal.

Với label \(a_i\):

\[
\left[P^\top D_AP\right]_{kk}
=
\sum_{i:a_i=k}d_{A,i}.
\]

Đặt:

\[
V_N=P^\top U_N,
\qquad
V_K=P^\top U_K.
\]

Khi đó:

\[
\boxed{
L
=
\operatorname{Diag}
\left(
P^\top d_A
\right)
+
V_NV_N^\top
+
\mu V_KV_K^\top.
}
\]

---

# 28. Tính RHS của codebook system

Ta cần:

\[
b=P^\top Aw-P^\top g.
\]

Tính:

\[
\boxed{
Aw
=
d_A\odot w
+
U_N(U_N^\top w)
+
\mu U_K(U_K^\top w).
}
\]

Sau đó:

\[
\boxed{
b
=
P^\top(Aw-g).
}
\]

Implementation dùng scatter-add theo labels.

Không materialize dense \(P\).

---

# 29. Codebook update pseudocode

```python
def exact_codebook_update(
    w,
    g,
    labels,
    d_nll,
    U_nll,
    d_kl,
    U_kl,
    mu,
    nu,
    K,
):
    dA = d_nll + mu * d_kl + nu

    diag_L = scatter_sum(dA, labels, dim_size=K)

    VN = scatter_sum_rows(U_nll, labels, dim_size=K)
    VK = scatter_sum_rows(U_kl, labels, dim_size=K)

    L = torch.diag(diag_L)
    L = L + VN @ VN.T
    L = L + mu * (VK @ VK.T)

    Aw = dA * w
    Aw = Aw + U_nll @ (U_nll.T @ w)
    Aw = Aw + mu * (U_kl @ (U_kl.T @ w))

    b = scatter_sum(Aw - g, labels, dim_size=K)

    solve active-cluster subsystem
    keep old codeword for empty clusters
    return updated codebook
```

---

# 30. Empty clusters và singular systems

Nếu cluster không có assigned coordinate:

- bỏ cluster đó khỏi active system;
- giữ nguyên codeword cũ;
- không reseed tùy ý trong monotone loop.

Nếu active system singular:

- dùng `torch.linalg.lstsq`;
- hoặc pseudoinverse;
- không thêm manual damping hyperparameter.

Sau update:

1. sort codebook;
2. remap labels;
3. reconstructed \(q\) phải không đổi.

---

# 31. Objective và constraint evaluation

NLL surrogate:

\[
\boxed{
J_N(e)
=
g^\top e
+
\frac12
\left[
\sum_i d_{N,i}e_i^2
+
\|U_N^\top e\|^2
\right].
}
\]

KL constraint:

\[
\boxed{
C_K(e)
=
\frac12
\left[
\sum_i d_{K,i}e_i^2
+
\|U_K^\top e\|^2
\right].
}
\]

Weight constraint:

\[
\boxed{
C_w(e)
=
\frac12\|e\|^2.
}
\]

Fixed-dual objective:

\[
\boxed{
J(e)
=
J_N(e)
+
\mu C_K(e)
+
\nu C_w(e)
}
\]

bỏ các dual constants.

---

# 32. Dual controller

Giữ nguyên:

\[
v_K
=
\frac{C_K}{\epsilon_K}-1,
\]

\[
v_w
=
\frac{C_w}{\epsilon_w}-1.
\]

Trace:

\[
\boxed{
\operatorname{tr}(D+UU^\top)
=
\sum_i d_i+\|U\|_F^2.
}
\]

Dùng trace này cho scale-aware dual controller.

Luôn giữ:

```text
best_feasible = q0
```

Candidate infeasible không được overwrite final output.

---

# 33. End-to-end pipeline

## Stage 1: Initialization

Run `sqllm`:

\[
q_0,\ labels_0,\ codebook_0.
\]

## Stage 2: NLL pass

Dùng ground-truth labels:

- collect signed \(g\);
- collect NLL sketch;
- finalize \(D_N,U_N\).

## Stage 3: KL pass

Dùng teacher pseudo-label score:

- collect KL sketch;
- finalize \(D_K,U_K\).

## Stage 4: Precompute

- \(r_N\);
- \(r_K\);
- trace scales;
- save cache.

## Stage 5: Quantization

Per row:

- initialize tại \(q_0\);
- compute budgets;
- run outer dual loop;
- inner Parallel MM assignment;
- exact codebook update;
- keep best feasible.

---

# 34. Tensor interfaces

Per module:

```text
weight:
    [d_out, n]

g:
    [d_out, n]

group_id:
    [d_out]

d_nll:
    [num_groups, n]

U_nll:
    [num_groups, n, rank_nll]

d_kl:
    [num_groups, n]

U_kl:
    [num_groups, n, rank_kl]

row_majorizer_nll:
    [num_groups, n]

row_majorizer_kl:
    [num_groups, n]

labels:
    [d_out, n]

codebooks:
    [d_out, K]

mu:
    [d_out]

nu:
    [d_out]
```

Không còn:

```text
H_nll:
    [num_groups, n, n]

H_kl:
    [num_groups, n, n]
```

---

# 35. Cache format

Recommended cache per module:

```text
signed_gradient.pt
nll_diag.pt
nll_lowrank.pt
kl_diag.pt
kl_lowrank.pt
group_ids.pt
metadata.json
```

Metadata phải chứa:

```text
model identifier/checkpoint hash
module name
weight shape
calibration dataset
number of valid tokens
token mask policy
loss reduction
curvature grouping
rank
sketch seed
dtype
number of KL probes
```

Không reuse cache nếu model, calibration hoặc normalization khác.

---

# 36. Numerical precision

Recommended:

- model forward: model dtype;
- \(g\): FP32 accumulation;
- sketch \(C,B,h_{\mathrm{diag}}\): FP32;
- eigendecomposition \(B\): FP32 hoặc FP64 khi debug;
- MM products: FP32/TF32 đã validate;
- codebook solve: FP32;
- objective checks: FP64 khi debug.

Không lưu \(D,U\) ở FP16 trước khi xác nhận scale ổn định.

---

# 37. Debug metrics

## NLL/KL collector

```text
total_valid_tokens
min/median/max sensitivity
trace_target_from_diag
effective_rank_B
min/max retained_eigenvalue
fraction_negative_residual_before_clamp
max_negative_residual_before_clamp
```

## Curvature representation

```text
sum_D
frobenius_U_squared
trace_D_plus_UUT
min_D
median_D
max_D
```

## Random-vector validation

Với random \(v\), nếu chạy một dense small-module reference:

```text
relative_error_Hv
relative_error_quadratic
relative_error_diagonal
```

## MM

```text
objective_before
objective_after_assignment
objective_after_codebook
num_label_changes
max_abs_target
max_abs_codebook
```

Invariant:

```text
objective_after_assignment <= objective_before
objective_after_codebook <= objective_after_assignment
```

---

# 38. Unit tests bắt buộc

## Test 1: PSD

\[
d_i\ge0.
\]

Với random \(v\):

\[
v^\top(D+UU^\top)v\ge0.
\]

## Test 2: Diagonal preservation

\[
\operatorname{diag}(D+UU^\top)
\approx
h_{\mathrm{diag}}.
\]

## Test 3: Streaming equivalence

Trên toy data:

- collect một batch;
- collect nhiều mini-batches;
- kết quả \(C,B,h_{\mathrm{diag}}\) phải khớp.

## Test 4: Normalization

Sum-loss và mean-loss đã undo scale phải khớp.

## Test 5: NLL/KL source correctness

NLL:

```text
uses ground-truth labels
```

KL:

```text
uses teacher pseudo-labels
does not use ground-truth labels
```

## Test 6: Majorizer

Với random \(v\):

\[
v^\top Mv
\ge
v^\top Av.
\]

## Test 7: MM monotonicity

One assignment step không tăng fixed-dual objective.

## Test 8: Exact codebook update

Codebook solve không tăng objective.

## Test 9: \(q_0\) feasibility

\[
C_K(q_0)=\epsilon_K,
\]

\[
C_w(q_0)=\epsilon_w.
\]

---

# 39. Performance notes

Dense curvature:

\[
O(Tn^2).
\]

Direct sketch:

\[
O(Tnr).
\]

Storage:

\[
O(nr+n+r^2).
\]

Solver Hessian product:

\[
O(Bnr).
\]

Không còn:

- dense Hessian write/read;
- dense \(n\times n\) GEMM trong MM;
- dense \(P^\top HP\) aggregation.

NLL pass nên collect \(g\) và \(D_N,U_N\) cùng lúc.

KL pass nên bắt đầu với một Fisher probe.

---

# 40. Những lỗi không được lặp lại

1. Không tính dense \(H\) rồi mới phân rã.
2. Không dùng \(E[\delta^2]E[xx^\top]\).
3. Không dùng gradient cuối cùng \(g\) để suy ra curvature.
4. Không average gradient các KL probes trước khi square.
5. Không dùng mean-loss gradient rồi square nếu chưa undo normalization.
6. Không dùng NLL gradients cho KL.
7. Không dùng KL score gradients làm signed \(g\).
8. Không thêm manual damping vào \(D\) chỉ để tránh lỗi số.
9. Không materialize dense \(P\).
10. Không claim exactness đối với dense GuidedQuant Hessian.
11. Không claim global optimum của discrete constrained problem.
12. Không overwrite best feasible bằng candidate infeasible.

---

# 41. Safe theoretical claims

Có thể claim:

1. \(D+UU^\top\) là PSD khi \(D\ge0\).
2. Collector trực tiếp approximate GuidedQuant-style weighted covariance.
3. Approximation không cần materialize dense Hessian.
4. Safe row majorizer majorizes approximate curvature.
5. Parallel MM monotonically giảm fixed-dual approximate objective.
6. Exact codebook update monotonically giảm fixed-label approximate objective.
7. Alternating assignment/codebook updates monotone với fixed duals.

Không claim:

1. \(D+UU^\top\) bằng exact Hessian.
2. \(D+UU^\top\) bằng dense GuidedQuant curvature.
3. global optimum của constrained scalar quantization.
4. strong duality cho discrete problem.
5. outer dual loop monotone trên cùng một objective.

---

# 42. Recommended implementation modules

```text
stats/
    collect_nll_gradient_and_sketch.py
    collect_kl_fisher_sketch.py
    finalize_diag_lowrank.py
    curvature_grouping.py

init/
    sqllm_adapter.py

solver/
    lowrank_curvature_ops.py
    parallel_mm_assignment.py
    exact_codebook_update_lowrank.py
    dual_controller.py

quant/
    method_a_quantizer.py

utils/
    batched_searchsorted.py
    scatter_ops.py
    curvature_cache.py
    objective_metrics.py
```

---

# 43. Final pseudocode

```text
INPUT:
    FP model
    calibration data
    bit-width b
    K = 2^b
    curvature groups
    sketch rank r
    sketch seed
    KL probes M

STAGE 1: SQ LLM INITIALIZATION
    Q0, labels0, codebooks0 = run_sqllm(...)

STAGE 2: NLL STATS
    initialize C_N, B_N, diag_N per group
    run ground-truth NLL forward/backward
    collect signed g
    collect grouped NLL sensitivity
    stream-update C_N, B_N, diag_N
    finalize D_N, U_N

STAGE 3: KL STATS
    initialize C_K, B_K, diag_K per group
    for probe in 1..M:
        sample pseudo-labels from teacher
        backward teacher log-prob score
        collect grouped squared score gradients
        stream-update C_K, B_K, diag_K
    finalize D_K, U_K

STAGE 4: PRECOMPUTE
    r_N = lowrank_row_majorizer(D_N, U_N)
    r_K = lowrank_row_majorizer(D_K, U_K)

FOR each curvature group:
    batch rows sharing D_N,U_N,D_K,U_K

    initialize:
        Q = Q0
        labels = labels0
        codebooks = codebooks0
        E0 = Q0 - W

        eps_K =
            0.5 * (
                sum(D_K * E0^2)
                + ||E0 @ U_K||^2
            )

        eps_w =
            0.5 * sum(E0^2)

        best_feasible = Q0

    FOR outer_iter:
        FOR inner_iter:
            E = Q - W

            EN =
                E * D_N
                + (E @ U_N) @ U_N.T

            EK =
                E * D_K
                + (E @ U_K) @ U_K.T

            S =
                G
                + EN
                + mu * EK
                + nu * E

            Mdiag =
                r_N
                + mu * r_K
                + nu

            target = Q - S / Mdiag

            labels_new =
                nearest_sorted_codeword(
                    target,
                    codebooks,
                    current_labels
                )

            Q = gather(codebooks, labels_new)

            codebooks =
                exact_codebook_update_lowrank(
                    W, G,
                    labels_new,
                    D_N, U_N,
                    D_K, U_K,
                    mu, nu
                )

            sort codebooks
            remap labels
            Q = gather(codebooks, labels)

            verify fixed-dual objective does not increase

            if labels stable:
                break

        compute KL and Euclidean constraints

        if feasible:
            compare NLL surrogate
            update best_feasible

        update mu, nu

    write best_feasible to model
```

---

# 44. One-line formula sheet

Curvature representation:

\[
\boxed{
H_N\approx D_N+U_NU_N^\top
}
\]

\[
\boxed{
H_K\approx D_K+U_KU_K^\top
}
\]

Streaming weighted sample:

\[
\boxed{
z_t=\sqrt{s_t}x_t
}
\]

Sketch:

\[
\boxed{
C=\frac1T\sum_t z_t(z_t^\top\Omega)
}
\]

\[
\boxed{
B=\frac1T\sum_t
(\Omega^\top z_t)(z_t^\top\Omega)
}
\]

Low rank:

\[
\boxed{
U=CV\Lambda^{-1/2}
}
\]

Diagonal:

\[
\boxed{
D=
\operatorname{Diag}
\left[
\operatorname{diag}(H)-\operatorname{diag}(UU^\top)
\right]_+
}
\]

Fixed-dual gradient:

\[
\boxed{
s
=
g
+
d_N\odot e
+
U_N(U_N^\top e)
+
\mu
\left[
d_K\odot e
+
U_K(U_K^\top e)
\right]
+
\nu e
}
\]

Majorizer:

\[
\boxed{
m
=
r_N+\mu r_K+\nu
}
\]

MM target:

\[
\boxed{
t=q-s\oslash m
}
\]

Assignment:

\[
\boxed{
q_i^+=\operatorname{NearestCodeword}(t_i)
}
\]

Codebook:

\[
\boxed{
(P^\top AP)c=P^\top Aw-P^\top g
}
\]

---

# 45. Kết luận

Phiên bản mới giữ nguyên toàn bộ ý tưởng chính của Method A:

- constrained end-loss formulation;
- signed NLL gradient;
- NLL và KL curvature tách biệt;
- `sqllm` initialization;
- trust-region budgets từ \(q_0\);
- Parallel MM assignment;
- exact codebook update;
- best feasible fallback.

Thay đổi duy nhất ở mức phương pháp là biểu diễn curvature:

\[
\boxed{
\text{dense grouped }H
\quad\longrightarrow\quad
\text{direct streaming }D+UU^\top.
}
\]

Điều này giảm mạnh chi phí statistics collection, memory và solver compute, trong khi vẫn giữ đúng nguồn end-loss gradient của NLL và teacher-score Fisher của KL.

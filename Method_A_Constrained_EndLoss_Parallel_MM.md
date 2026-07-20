# Method A: Constrained End-Loss Scalar Quantization
## GuidedQuant-style dense grouped curvature + Parallel MM assignment + exact codebook update

> **Mục tiêu của tài liệu:** đặc tả đủ chi tiết để triển khai một phương pháp **weight-only non-uniform scalar quantization** trên GPU.
>
> Mỗi scalar weight được thay bằng một trong \(K\) scalar codewords. Đây **không phải vector quantization**.
>
> Phương pháp dùng:
>
> - end NLL làm objective cần tối ưu;
> - KL giữa model gốc và model quantized làm ràng buộc giữ hành vi đầu ra;
> - một Euclidean trust region để ngăn perturbation weight quá xa;
> - hai curvature khác nhau: \(H_{\mathrm{NLL}}\) và \(H_{\mathrm{KL}}\);
> - cách xấp xỉ Hessian/Fisher dạng dense grouped block lấy cảm hứng từ GuidedQuant;
> - initialization từ nhánh `sqllm` trong repository GuidedQuant;
> - assignment bằng **Parallel MM**, không dùng coordinate descent của LNQ;
> - codebook update bằng nghiệm chính xác của một hệ tuyến tính \(K\times K\).

---

# 1. Reference implementation

GuidedQuant repository:

https://github.com/snu-mllab/GuidedQuant

Paper:

https://openreview.net/forum?id=ZawsPjlIGu

Trong repository GuidedQuant:

- `scripts/run_sqllm.sh` là pipeline scalar quantization dùng làm reference/init trong tài liệu này;
- GuidedQuant thu weight gradients và activation gradients từ end loss;
- curvature block của một output channel có cấu trúc dạng

\[
H_j
=
X^\top
\operatorname{Diag}(s_j)
X,
\]

và nhiều output channels có thể được gom nhóm rồi average curvature để giảm chi phí.

**Quan trọng:** phương pháp ở tài liệu này chỉ mượn **cấu trúc tính curvature/grouping** và có thể reuse code collector của GuidedQuant. Không dùng LNQ coordinate descent làm assignment solver.

---

# 2. Scope và notation

Xét một linear layer của PyTorch:

\[
Y = XW^\top,
\]

với:

- \(W\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}}\);
- mỗi row \(w_j\in\mathbb R^{d_{\mathrm{in}}}\) là weight vector của output channel \(j\);
- \(X\in\mathbb R^{T\times d_{\mathrm{in}}}\) là input activations thu trên calibration tokens;
- \(T\) là tổng số valid calibration tokens dùng cho statistics.

Ta quantize từng row:

\[
w \in \mathbb R^n,
\qquad
n=d_{\mathrm{in}}.
\]

Một row sau quantization là:

\[
q_i\in\mathcal C,
\]

với scalar codebook:

\[
\mathcal C=\{c_1,\dots,c_K\}.
\]

Đặt quantization error:

\[
e=q-w.
\]

Trong implementation hiện tại nên hiểu:

- **quantization group**: một output row \(w_j\);
- **curvature group**: một nhóm output rows \(J_k\) cùng dùng chung \(H_{\mathrm{NLL},k}\) và \(H_{\mathrm{KL},k}\).

Như vậy codebook có thể riêng cho từng row, trong khi curvature được share giữa nhiều rows để giảm memory và compute.

> GuidedQuant paper thường viết \(W\in\mathbb R^{d_{\mathrm{in}}\times d_{\mathrm{out}}}\), tức output channel là một **column**. PyTorch `nn.Linear.weight` có shape \([d_{\mathrm{out}},d_{\mathrm{in}}]\), nên trong code của ta output channel là một **row**. Phải thống nhất orientation này khi port công thức.

---

# 3. Bài toán gốc

Ta muốn model quantized có NLL thấp, nhưng không được làm thay đổi phân phối đầu ra quá nhiều.

Bài toán gốc:

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

Trong đó:

- \(w\): full-precision weight row;
- \(q\): scalar-quantized weight row;
- \(\mathcal Q_K\): tập các vectors mà mỗi phần tử chỉ được nhận một trong \(K\) codewords;
- \(L_{\mathrm{NLL}}\): end NLL trên calibration data;
- KL constraint: giữ output distribution của model quantized gần model gốc;
- Euclidean trust region: giữ perturbation weight nằm trong vùng local, đồng thời bảo vệ các flat/null directions mà KL curvature có thể rất nhỏ.

---

# 4. Taylor expansion của NLL

Đặt:

\[
e=q-w.
\]

Khai triển NLL quanh model gốc:

\[
L_{\mathrm{NLL}}(w+e)
\approx
L_{\mathrm{NLL}}(w)
+
g^\top e
+
\frac12 e^\top H_{\mathrm{NLL}}e,
\]

với:

\[
\boxed{
g=
\nabla_w L_{\mathrm{NLL}}(w)
}
\]

và exact Hessian:

\[
\boxed{
H_{\mathrm{NLL}}
=
\nabla_w^2 L_{\mathrm{NLL}}(w).
}
\]

Bỏ hằng số \(L_{\mathrm{NLL}}(w)\), local NLL objective là:

\[
\boxed{
g^\top e
+
\frac12 e^\top H_{\mathrm{NLL}}e.
}
\]

Ý nghĩa:

- \(g\) giữ **signed first-order direction**: perturbation nào có xu hướng tăng/giảm NLL;
- \(H_{\mathrm{NLL}}\) mô tả local curvature của NLL.

---

# 5. Taylor expansion của KL

Ta có:

\[
D_{\mathrm{KL}}
\left(
p_w
\|
p_{w+e}
\right)
=
\mathbb E_{y\sim p_w}
\left[
\log p_w(y)
-
\log p_{w+e}(y)
\right].
\]

Tại \(e=0\):

\[
D_{\mathrm{KL}}(p_w\|p_w)=0.
\]

Gradient bậc nhất bằng 0:

\[
\nabla_e
D_{\mathrm{KL}}
(p_w\|p_{w+e})
\Big|_{e=0}
=
0.
\]

Do đó KL bắt đầu từ bậc hai:

\[
D_{\mathrm{KL}}
\left(
p_w
\|
p_{w+e}
\right)
\approx
\frac12
e^\top
H_{\mathrm{KL}}
e.
\]

Với:

\[
\boxed{
H_{\mathrm{KL}}
=
\nabla_e^2
D_{\mathrm{KL}}
(p_w\|p_{w+e})
\Big|_{e=0}.
}
\]

Dùng Fisher identity:

\[
\boxed{
H_{\mathrm{KL}}
=
\mathbb E_{y\sim p_w}
\left[
\nabla_w \log p_w(y)
\nabla_w \log p_w(y)^\top
\right].
}
\]

Do đó ta **không cần thực sự tạo model quantized và tính KL** để collect \(H_{\mathrm{KL}}\).

Ta có thể:

1. chạy teacher/model gốc;
2. lấy output distribution \(p_w\);
3. sample pseudo-label \(y\sim p_w\);
4. backward \(\log p_w(y)\);
5. dùng squared score gradients để xây Fisher.

Đây là nguồn đúng của \(H_{\mathrm{KL}}\).

---

# 6. Phải phân biệt rõ \(g\), \(H_{\mathrm{NLL}}\), \(H_{\mathrm{KL}}\)

Đây là phần rất quan trọng khi code.

## 6.1. Gradient \(g\)

Nguồn:

\[
\boxed{
\text{ground-truth NLL}
}
\]

Cụ thể:

\[
L_{\mathrm{NLL}}
=
-\frac1T
\sum_{t=1}^T
\log
p_w(y_t^{\mathrm{true}}|x_{\le t}).
\]

Sau đó:

\[
\boxed{
g
=
\nabla_w L_{\mathrm{NLL}}.
}
\]

\(g\) là signed gradient.

Không square \(g\).

Không lấy \(g\) từ KL.

---

## 6.2. \(H_{\mathrm{NLL}}\)

Về lý thuyết:

\[
H_{\mathrm{NLL}}
=
\nabla_w^2 L_{\mathrm{NLL}}.
\]

Nhưng exact Hessian rất đắt và có thể indefinite.

Trong implementation ta dùng một **GuidedQuant-style PSD empirical-Fisher/GGN approximation** được xây từ **ground-truth NLL end-loss gradients**.

Nguồn upstream loss vẫn là:

\[
\boxed{
-\log p_w(y_t^{\mathrm{true}}|x_{\le t}).
}
\]

---

## 6.3. \(H_{\mathrm{KL}}\)

Nguồn:

\[
\boxed{
\text{teacher/model output distribution}
}
\]

Không dùng ground-truth label.

Một Fisher probe dùng:

\[
y_t^{(m)}
\sim
p_w(\cdot|x_{\le t})
\]

và score:

\[
\log
p_w
(y_t^{(m)}|x_{\le t}).
\]

Sau đó backward score này để lấy layer-output gradients.

Nếu dùng \(M\) Monte-Carlo Fisher probes:

\[
H_{\mathrm{KL}}
\approx
\frac1M
\sum_{m=1}^M
H_{\mathrm{KL}}^{(m)}.
\]

\(M\) là compute/variance trade-off, không phải hệ số của loss.

---

# 7. Tại sao hai Hessian không được trộn nguồn

Trong theory:

\[
H_{\mathrm{NLL}}
\]

là curvature của objective NLL.

Trong khi:

\[
H_{\mathrm{KL}}
\]

là curvature của KL constraint.

Ở final-logit space, softmax cross entropy và KL có cùng local matrix:

\[
\operatorname{Diag}(p)-pp^\top.
\]

Nhưng ở weight space, exact \(H_{\mathrm{NLL}}\) nói chung có thêm network second-derivative terms, trong khi KL tại teacher = student có first derivative bằng 0.

Vì vậy không nên nói:

\[
H_{\mathrm{NLL}}=H_{\mathrm{KL}}
\]

một cách mặc định.

Trong implementation hiện tại, ta dùng **cùng một structural approximation kiểu GuidedQuant** cho cả hai, nhưng upstream gradient khác nhau:

\[
\boxed{
H_{\mathrm{NLL}}
\leftarrow
\text{ground-truth NLL gradients}
}
\]

\[
\boxed{
H_{\mathrm{KL}}
\leftarrow
\text{teacher score/log-prob gradients}.
}
\]

---

# 8. Local constrained problem

Sau Taylor approximation:

\[
\boxed{
\min_{q\in\mathcal Q_K}
g^\top e
+
\frac12
e^\top
H_{\mathrm{NLL}}
e
}
\]

subject to:

\[
\boxed{
\frac12
e^\top
H_{\mathrm{KL}}
e
\le
\epsilon_{\mathrm{KL}}
}
\]

và:

\[
\boxed{
\frac12
\|e\|_2^2
\le
\epsilon_w.
}
\]

Đây là formulation chính của Method A.

---

# 9. Initialization: dùng `sqllm` trong GuidedQuant

Initialization chốt cho implementation đầu tiên:

\[
\boxed{
q_0
=
Q_{\mathrm{sqllm}}(w)
}
\]

trong đó `sqllm` là pipeline scalar quantization trong repository GuidedQuant:

https://github.com/snu-mllab/GuidedQuant

Script entry:

```bash
scripts/run_sqllm.sh
```

## Quy tắc quan trọng

Dùng output quantized weight của `sqllm` làm:

1. initial quantized state;
2. reference point để tự động xác định trust-region budgets.

Không chạy thêm một H-only optimizer trước khi lấy budget.

Không dùng:

\[
w-H^{-1}g
\]

làm initialization.

Không dùng \(H^{-1}g\) tự do.

Initialization phải là một quantized solution thực sự ở đúng bit-width.

Đặt:

\[
e_0=q_0-w.
\]

Sau đó:

\[
\boxed{
\epsilon_{\mathrm{KL}}
=
\frac12
e_0^\top
H_{\mathrm{KL}}
e_0
}
\]

và:

\[
\boxed{
\epsilon_w
=
\frac12
\|e_0\|_2^2.
}
\]

Như vậy \(q_0\) luôn là feasible reference point.

Ý nghĩa của optimization sau đó:

- tìm quantized solution có local NLL tốt hơn;
- không cho KL/Fisher damage vượt quá mức của \(q_0\);
- không cho Euclidean displacement vượt quá mức của \(q_0\).

Nếu quantization group là một row, budgets nên được tính **per row**:

\[
\epsilon_{\mathrm{KL},j}
=
\frac12
e_{0,j}^\top
H_{\mathrm{KL},k(j)}
e_{0,j},
\]

\[
\epsilon_{w,j}
=
\frac12
\|e_{0,j}\|_2^2.
\]

Ở đây \(k(j)\) là curvature group chứa output row \(j\).

> Lưu ý: có thể reuse `sqllm` implementation để tạo \(q_0\), nhưng không được reuse nhầm curvature nội bộ của `sqllm` rồi gọi nó là \(H_{\mathrm{NLL}}\) hoặc \(H_{\mathrm{KL}}\). Hai curvature của Method A phải được collect riêng từ đúng loss/source như Sections 6 và 12.

---

# 10. Lagrangian form

Dùng dual variables:

\[
\mu\ge0,
\qquad
\nu\ge0.
\]

Lagrangian:

\[
\mathcal L(e,\mu,\nu)
=
g^\top e
+
\frac12 e^\top H_{\mathrm{NLL}}e
+
\mu
\left(
\frac12 e^\top H_{\mathrm{KL}}e
-
\epsilon_{\mathrm{KL}}
\right)
+
\nu
\left(
\frac12\|e\|_2^2
-
\epsilon_w
\right).
\]

Với fixed \(\mu,\nu\), bỏ các hằng số không phụ thuộc \(q\):

\[
\boxed{
J(e)
=
g^\top e
+
\frac12 e^\top A e
}
\]

với:

\[
\boxed{
A
=
H_{\mathrm{NLL}}
+
\mu H_{\mathrm{KL}}
+
\nu I.
}
\]

Đây là objective mà inner solver tối ưu.

Vì implementation dùng PSD Fisher/GGN approximations:

\[
H_{\mathrm{NLL}}\succeq0,
\qquad
H_{\mathrm{KL}}\succeq0,
\]

nên:

\[
A\succeq0.
\]

Nếu:

\[
\nu>0,
\]

thì:

\[
A\succ0.
\]

---

# 11. Scalar quantization representation

Với một row:

\[
w\in\mathbb R^n.
\]

Codebook:

\[
c=
[c_1,\dots,c_K]^\top.
\]

Assignment matrix:

\[
P\in\{0,1\}^{n\times K},
\]

mỗi row của \(P\) có đúng một phần tử bằng 1.

Khi đó:

\[
\boxed{
q=Pc.
}
\]

Trong code thực tế không nhất thiết materialize dense one-hot \(P\). Có thể lưu labels:

```text
labels[i] ∈ {0, ..., K-1}
```

và:

```python
q = codebook[labels]
```

---

# 12. GuidedQuant-style Hessian collection

## 12.1. Cấu trúc chung

Với một linear layer:

\[
Z=XW^\top.
\]

Xét output channel \(j\).

Gọi:

\[
\delta_{t,j}
=
\frac{\partial \ell_t}{\partial Z_{t,j}},
\]

trong đó \(\ell_t\) là end-loss scalar signal cho token \(t\).

Gradient của loss theo row weight \(w_j\):

\[
\nabla_{w_j}\ell_t
=
\delta_{t,j}x_t.
\]

Outer product:

\[
\nabla_{w_j}\ell_t
\nabla_{w_j}\ell_t^\top
=
\delta_{t,j}^2
x_tx_t^\top.
\]

Average trên calibration tokens:

\[
\boxed{
H_j
\approx
\frac1T
\sum_{t=1}^T
\delta_{t,j}^2
x_tx_t^\top.
}
\]

Dạng ma trận:

\[
\boxed{
H_j
=
\frac1T
X^\top
\operatorname{Diag}(s_j)
X,
}
\]

với:

\[
s_{t,j}=\delta_{t,j}^2.
\]

Đây là cấu trúc cần reuse từ GuidedQuant.

---

## 12.2. Group output channels như GuidedQuant

Không lưu một dense matrix \(d_{\mathrm{in}}\times d_{\mathrm{in}}\) cho mọi output channel.

Chia output channels thành các groups:

\[
J_1,\dots,J_G.
\]

Với group \(k\):

\[
\bar s_{t,k}
=
\frac1{|J_k|}
\sum_{j\in J_k}
s_{t,j}.
\]

Sau đó:

\[
\boxed{
H_k
=
\frac1T
X^\top
\operatorname{Diag}(\bar s_k)
X.
}
\]

Mọi output rows:

\[
j\in J_k
\]

dùng chung:

\[
H_k.
\]

Method A cần hai bộ:

\[
\boxed{
H_{\mathrm{NLL},k}
}
\]

và:

\[
\boxed{
H_{\mathrm{KL},k}.
}
\]

Cùng cấu trúc, khác nguồn \(\delta\).

---

# 13. Cách collect \(g\) và \(H_{\mathrm{NLL}}\)

## 13.1. Ground-truth NLL

Với mỗi valid calibration token:

\[
\ell_t^{\mathrm{NLL}}
=
-\log
p_w
(y_t^{\mathrm{true}}|x_{\le t}).
\]

Global NLL:

\[
L_{\mathrm{NLL}}
=
\frac1T
\sum_t
\ell_t^{\mathrm{NLL}}.
\]

Weight gradient:

\[
\boxed{
g
=
\nabla_w L_{\mathrm{NLL}}.
}
\]

Đây là signed gradient dùng trực tiếp trong objective.

---

## 13.2. NLL curvature signal

Tại một linear layer:

\[
\delta_{t,j}^{\mathrm{NLL}}
=
\frac{\partial
\ell_t^{\mathrm{NLL}}
}
{\partial Z_{t,j}}.
\]

Sau đó:

\[
s_{t,j}^{\mathrm{NLL}}
=
\left(
\delta_{t,j}^{\mathrm{NLL}}
\right)^2.
\]

Group average:

\[
\bar s_{t,k}^{\mathrm{NLL}}
=
\frac1{|J_k|}
\sum_{j\in J_k}
\left(
\delta_{t,j}^{\mathrm{NLL}}
\right)^2.
\]

Cuối cùng:

\[
\boxed{
H_{\mathrm{NLL},k}
=
\frac1T
X^\top
\operatorname{Diag}
\left(
\bar s_k^{\mathrm{NLL}}
\right)
X.
}
\]

---

## 13.3. Normalization cực kỳ quan trọng

Không được vô tình lấy gradient của **mean loss**, square nó rồi lại coi như mean squared per-token gradient.

Nếu:

\[
L
=
\frac1T
\sum_t
\ell_t,
\]

thì autograd của mean loss có factor \(1/T\).

Nếu hook thu:

\[
\frac{\partial L}{\partial Z_{t,j}}
=
\frac1T
\frac{\partial\ell_t}{\partial Z_{t,j}},
\]

rồi square trực tiếp, ta nhận thêm factor:

\[
\frac1{T^2},
\]

không phải curvature mong muốn:

\[
\frac1T
\sum_t
\delta_t^2x_tx_t^\top.
\]

Vì vậy collector phải nhất quán.

Một cách:

1. backward `loss_sum`, không phải `loss_mean`;
2. lấy per-token \(\delta_t\);
3. accumulate

\[
\sum_t
\delta_t^2x_tx_t^\top;
\]

4. cuối cùng chia cho tổng valid tokens \(T\).

Hoặc nếu backward mean loss thì phải undo đúng reduction scaling trước khi square.

Weight gradient \(g\) cuối cùng vẫn nên là gradient của **mean NLL/token**:

\[
g
=
\frac1T
\nabla_w
\sum_t
\ell_t.
\]

---

# 14. Cách collect \(H_{\mathrm{KL}}\)

## 14.1. Không cần trực tiếp tính KL

Teacher/model gốc sinh:

\[
p_t
=
p_w(\cdot|x_{\le t}).
\]

Sample pseudo-label:

\[
y_t^{(m)}
\sim
p_t.
\]

Định nghĩa score:

\[
a_t^{(m)}
=
\log
p_w
(y_t^{(m)}|x_{\le t}).
\]

Tại linear layer:

\[
\delta_{t,j}^{\mathrm{KL},(m)}
=
\frac{\partial
a_t^{(m)}
}
{\partial Z_{t,j}}.
\]

Fisher sensitivity:

\[
s_{t,j}^{\mathrm{KL}}
\approx
\frac1M
\sum_{m=1}^M
\left(
\delta_{t,j}^{\mathrm{KL},(m)}
\right)^2.
\]

Sau output-channel grouping:

\[
\bar s_{t,k}^{\mathrm{KL}}
=
\frac1{|J_k|}
\sum_{j\in J_k}
s_{t,j}^{\mathrm{KL}}.
\]

Cuối cùng:

\[
\boxed{
H_{\mathrm{KL},k}
=
\frac1T
X^\top
\operatorname{Diag}
\left(
\bar s_k^{\mathrm{KL}}
\right)
X.
}
\]

Đây là Monte-Carlo approximation của KL Fisher block.

---

## 14.2. Một probe

Prototype đầu tiên có thể dùng:

\[
M=1.
\]

Một pseudo-label per token cho unbiased Monte-Carlo Fisher estimate.

Tăng \(M\) chỉ để giảm variance.

Không dùng \(y_t^{\mathrm{true}}\) trong KL collector.

Không dùng NLL activation gradients thay cho KL Fisher gradients.

---

# 15. Streaming Hessian accumulation

Không cần giữ toàn bộ \(X\) và \(\delta\) của calibration set trong GPU memory.

Với mỗi calibration batch \(b\):

\[
X_b\in\mathbb R^{T_b\times n}.
\]

Với group sensitivity:

\[
s_{b,k}\in\mathbb R^{T_b}.
\]

Accumulate:

\[
\boxed{
H_k
\mathrel{+}=
X_b^\top
\left(
s_{b,k}[:,None]\odot X_b
\right).
}
\]

Cuối calibration:

\[
H_k
\leftarrow
\frac{H_k}{T}.
\]

Làm riêng cho:

\[
H_{\mathrm{NLL},k}
\]

và:

\[
H_{\mathrm{KL},k}.
\]

Recommended:

- accumulation ở FP32;
- process từng layer/module;
- save curvature ra disk/CPU sau khi hoàn tất module nếu GPU memory hạn chế;
- quantization xử lý từng module/layer, không cần load Hessian của toàn model cùng lúc.

---

# 16. Optional factor form \(R^\top R\)

Vì:

\[
H
=
\frac1T
X^\top
\operatorname{Diag}(s)
X,
\]

đặt:

\[
\boxed{
R
=
\operatorname{Diag}
\left(
\sqrt{\frac{s}{T}}
\right)
X.
}
\]

Thì:

\[
\boxed{
H=R^\top R.
}
\]

Tuy nhiên với rất nhiều calibration tokens, lưu \(R\) có thể tốn memory hơn lưu dense \(H\).

Do đó implementation mặc định có thể:

- stream-accumulate dense grouped \(H\);
- chỉ dùng \(R\)-form nếu representation này thực sự rẻ hơn trong một setting cụ thể.

---

# 17. Parallel MM assignment

Ta giải fixed-dual objective:

\[
J(e)
=
g^\top e
+
\frac12e^\top Ae.
\]

Tại iteration \(t\):

\[
q^{(t)},
\qquad
e^{(t)}=q^{(t)}-w.
\]

Gradient theo \(q\):

\[
\boxed{
s^{(t)}
=
g
+
Ae^{(t)}.
}
\]

Xét candidate:

\[
q^{(t+1)}
=
q^{(t)}
+
\Delta.
\]

Vì \(J\) là quadratic:

\[
J(q^{(t)}+\Delta)
=
J(q^{(t)})
+
s^{(t)\top}\Delta
+
\frac12
\Delta^\top
A
\Delta.
\]

Vấn đề:

\[
\Delta^\top A\Delta
\]

couple tất cả coordinates.

Ta cần diagonal majorizer để update mọi coordinates song song.

---

# 18. Gershgorin / absolute-row-sum majorizer

Với một symmetric PSD matrix \(H\), định nghĩa:

\[
r_i(H)
=
\sum_j
|H_{ij}|.
\]

Đặt:

\[
R_H
=
\operatorname{Diag}
(r(H)).
\]

Ta có:

\[
\boxed{
R_H\succeq H.
}
\]

Với:

\[
A
=
H_{\mathrm{NLL}}
+
\mu H_{\mathrm{KL}}
+
\nu I,
\]

precompute:

\[
r_i^{\mathrm{NLL}}
=
\sum_j
|
(H_{\mathrm{NLL}})_{ij}
|,
\]

\[
r_i^{\mathrm{KL}}
=
\sum_j
|
(H_{\mathrm{KL}})_{ij}
|.
\]

Chọn:

\[
\boxed{
m_i
=
r_i^{\mathrm{NLL}}
+
\mu r_i^{\mathrm{KL}}
+
\nu.
}
\]

Và:

\[
M
=
\operatorname{Diag}(m).
\]

Khi đó:

\[
\boxed{
M\succeq A.
}
\]

Lưu ý:

\[
r^{\mathrm{NLL}}
+
\mu r^{\mathrm{KL}}
+
\nu
\]

là một **safe majorizer**.

Không cần materialize:

\[
A
\]

rồi tính lại absolute row sums mỗi khi \(\mu,\nu\) thay đổi.

---

# 19. MM surrogate

Vì:

\[
M\succeq A,
\]

nên:

\[
\Delta^\top A\Delta
\le
\Delta^\top M\Delta.
\]

Do đó:

\[
J(q^{(t)}+\Delta)
\le
J(q^{(t)})
+
s^{(t)\top}\Delta
+
\frac12
\Delta^\top M\Delta.
\]

Vì \(M\) diagonal:

\[
\boxed{
J(q^{(t)}+\Delta)
\le
J(q^{(t)})
+
\sum_i
\left[
s_i^{(t)}\Delta_i
+
\frac12
m_i\Delta_i^2
\right].
}
\]

Surrogate tách hoàn toàn theo từng coordinate.

---

# 20. Continuous MM target

Với:

\[
m_i>0,
\]

minimize:

\[
s_i\Delta_i
+
\frac12m_i\Delta_i^2.
\]

Nghiệm continuous:

\[
\Delta_i^\star
=
-\frac{s_i}{m_i}.
\]

Do:

\[
\Delta_i=q_i^{\mathrm{new}}-q_i^{\mathrm{old}},
\]

target:

\[
\boxed{
t_i
=
q_i^{(t)}
-
\frac{s_i^{(t)}}{m_i}.
}
\]

---

# 21. Discrete scalar assignment

Vì:

\[
q_i^{(t+1)}
\in
\mathcal C,
\]

ta chọn:

\[
\boxed{
q_i^{(t+1)}
=
\operatorname*{argmin}_{c\in\mathcal C}
|c-t_i|.
}
\]

Tương đương:

\[
\boxed{
a_i^{(t+1)}
=
\operatorname{NearestCodewordIndex}(t_i).
}
\]

Tất cả \(i\) được update **đồng thời**.

Đây là:

\[
\boxed{
\text{Parallel MM assignment}
}
\]

không phải cyclic coordinate descent.

---

# 22. Tie-breaking

Nếu hai codewords cách target bằng nhau:

\[
|c_a-t_i|
=
|c_b-t_i|,
\]

ưu tiên:

1. current codeword nếu nó nằm trong tie;
2. nếu không, chọn rule deterministic cố định.

Mục tiêu:

- tránh label oscillation;
- reproducible;
- giữ monotonicity.

---

# 23. Trường hợp \(m_i=0\)

Không thêm damping toán học chỉ để tránh chia 0.

Nếu:

\[
m_i=0,
\]

thì coordinate đó không có quadratic curvature trong majorizer.

Surrogate chỉ còn:

\[
s_i\Delta_i.
\]

Do đó solve trực tiếp trên finite codebook:

- nếu \(s_i>0\): chọn codeword nhỏ nhất;
- nếu \(s_i<0\): chọn codeword lớn nhất;
- nếu \(s_i=0\): giữ current codeword.

Trong code vẫn có thể dùng machine epsilon chỉ để chống lỗi floating-point, nhưng không coi nó là method damping hyperparameter.

---

# 24. Fixed-dual monotonicity của assignment

MM đảm bảo:

\[
J(q^{(t+1)})
\le
\operatorname{MM}
(q^{(t+1)}|q^{(t)})
\]

và vì \(q^{(t+1)}\) minimize surrogate trên scalar codebook:

\[
\operatorname{MM}
(q^{(t+1)}|q^{(t)})
\le
\operatorname{MM}
(q^{(t)}|q^{(t)}).
\]

Surrogate tight tại current point:

\[
\operatorname{MM}
(q^{(t)}|q^{(t)})
=
J(q^{(t)}).
\]

Vì vậy:

\[
\boxed{
J(q^{(t+1)})
\le
J(q^{(t)}).
}
\]

Guarantee này áp dụng cho **fixed \(\mu,\nu\) và fixed codebook**.

---

# 25. Exact codebook update

Giữ assignments \(P\) cố định.

Ta có:

\[
q=Pc.
\]

Objective:

\[
J(c)
=
g^\top(Pc-w)
+
\frac12
(Pc-w)^\top
A
(Pc-w).
\]

Đạo hàm theo \(c\):

\[
\nabla_cJ
=
P^\top g
+
P^\top A(Pc-w).
\]

Set bằng 0:

\[
P^\top g
+
P^\top APc
-
P^\top Aw
=
0.
\]

Do đó:

\[
\boxed{
(P^\top AP)c
=
P^\top Aw
-
P^\top g.
}
\]

Đặt:

\[
\boxed{
L
=
P^\top AP
}
\]

và:

\[
\boxed{
b
=
P^\top Aw
-
P^\top g.
}
\]

Codebook update là:

\[
\boxed{
Lc=b.
}
\]

Trong code:

```python
c = torch.linalg.solve(L, b)
```

nếu \(L\) nonsingular.

Không tính explicit matrix inverse.

---

# 26. Expand exact codebook system

Vì:

\[
A
=
H_{\mathrm{NLL}}
+
\mu H_{\mathrm{KL}}
+
\nu I,
\]

ta có:

\[
\boxed{
L
=
P^\top H_{\mathrm{NLL}}P
+
\mu
P^\top H_{\mathrm{KL}}P
+
\nu
P^\top P.
}
\]

Và:

\[
\boxed{
b
=
P^\top H_{\mathrm{NLL}}w
+
\mu
P^\top H_{\mathrm{KL}}w
+
\nu
P^\top w
-
P^\top g.
}
\]

Sau đó solve:

\[
\boxed{
\left[
P^\top H_{\mathrm{NLL}}P
+
\mu P^\top H_{\mathrm{KL}}P
+
\nu P^\top P
\right]c
=
P^\top H_{\mathrm{NLL}}w
+
\mu P^\top H_{\mathrm{KL}}w
+
\nu P^\top w
-
P^\top g.
}
\]

Hệ chỉ có kích thước:

\[
K\times K.
\]

Với 3-bit:

\[
K=8.
\]

Với 4-bit:

\[
K=16.
\]

---

# 27. Nếu dùng factor form

Nếu:

\[
H_{\mathrm{NLL}}
=
R_N^\top R_N
\]

và:

\[
H_{\mathrm{KL}}
=
R_K^\top R_K,
\]

thì:

\[
P^\top H_{\mathrm{NLL}}P
=
(R_NP)^\top(R_NP),
\]

\[
P^\top H_{\mathrm{KL}}P
=
(R_KP)^\top(R_KP).
\]

RHS:

\[
P^\top H_{\mathrm{NLL}}w
=
(R_NP)^\top(R_Nw),
\]

\[
P^\top H_{\mathrm{KL}}w
=
(R_KP)^\top(R_Kw).
\]

Nên:

\[
\boxed{
L
=
(R_NP)^\top(R_NP)
+
\mu
(R_KP)^\top(R_KP)
+
\nu P^\top P
}
\]

và:

\[
\boxed{
b
=
(R_NP)^\top(R_Nw)
+
\mu
(R_KP)^\top(R_Kw)
+
\nu P^\top w
-
P^\top g.
}
\]

---

# 28. Empty clusters và singular codebook system

Nếu một codeword không được assign weight nào, column tương ứng của \(P\) là zero.

Recommended:

1. xác định active clusters;
2. solve hệ chỉ trên active clusters;
3. giữ nguyên giá trị cũ của empty codewords;
4. không arbitrary reseed codeword bên trong fixed-dual monotone loop.

Nếu hệ active vẫn singular:

- dùng `torch.linalg.lstsq`, hoặc
- Moore-Penrose pseudoinverse.

Không cần thêm một damping hyperparameter chỉ để làm codebook solve invertible.

Nếu:

\[
\nu>0
\]

và mọi active clusters đều có phần tử, \(P^\top AP\) thường SPD.

---

# 29. Sort codebook sau update

Parallel nearest-codeword search hiệu quả nhất nếu codebook sorted.

Sau exact codebook update:

1. sort codebook;
2. remap labels theo permutation tương ứng;
3. đảm bảo reconstructed \(q\) không đổi.

Sorting/remapping chỉ đổi index semantics, không đổi objective.

---

# 30. Alternating inner solver

Với fixed \(\mu,\nu\):

1. Parallel MM assignment;
2. Exact codebook update;
3. repeat.

Cả hai bước đều không làm tăng fixed-dual objective:

\[
J.
\]

Do đó alternating solver monotone:

\[
\boxed{
J^{(t+1)}
\le
J^{(t)}.
}
\]

Stop khi:

- labels không đổi;
- hoặc objective improvement dưới numerical tolerance;
- hoặc đạt compute budget.

---

# 31. GPU-parallel implementation

Giả sử một curvature group có \(B\) output rows.

Stack:

\[
W\in\mathbb R^{B\times n},
\]

\[
Q\in\mathbb R^{B\times n},
\]

\[
G\in\mathbb R^{B\times n}.
\]

Mỗi row \(b\) có duals:

\[
\mu_b,\nu_b.
\]

Error:

\[
E=Q-W.
\]

Hai dense GEMMs:

\[
E_N
=
E
H_{\mathrm{NLL}},
\]

\[
E_K
=
E
H_{\mathrm{KL}}.
\]

Gradient của fixed-dual objective:

\[
\boxed{
S
=
G
+
E_N
+
\mu[:,None]\odot E_K
+
\nu[:,None]\odot E.
}
\]

Precomputed row-sums:

\[
r_N\in\mathbb R^n,
\qquad
r_K\in\mathbb R^n.
\]

Majorizer:

\[
\boxed{
M
=
r_N[None,:]
+
\mu[:,None]r_K[None,:]
+
\nu[:,None].
}
\]

Targets:

\[
\boxed{
T
=
Q
-
S\oslash M.
}
\]

Sau đó nearest-codeword cho toàn bộ:

\[
B\times n
\]

weights song song.

---

# 32. Nearest-codeword search

Không materialize tensor:

\[
[B,n,K]
\]

nếu \(K\) lớn.

Với sorted codebook từng row:

1. dùng batched `searchsorted`;
2. lấy left/right neighbor;
3. compare distance;
4. tie-break về current label.

Complexity assignment lookup:

\[
O(Bn\log K).
\]

Với \(K=8\) hoặc \(16\), brute-force GPU cũng có thể nhanh; tuy nhiên implementation nên giữ API hỗ trợ `searchsorted` để scale tới \(K\) lớn hơn.

---

# 33. Codebook update và dense Hessian: performance note

Assignment MM rất GPU-friendly vì mỗi iteration chủ yếu là:

- two dense GEMMs;
- elementwise operations;
- batched nearest-codeword search.

Exact codebook update với dense Hessian cần cẩn thận hơn.

Naive per-row computation của:

\[
P^\top HP
\]

có thể trở thành bottleneck nếu làm nhiều lần cho mọi output row.

Recommended engineering:

1. batch rows cùng curvature group;
2. dùng codebook update ít hơn assignment updates nếu profiling cho thấy đây là bottleneck;
3. reuse intermediate products khi labels không đổi;
4. exploit small \(K\);
5. dùng custom CUDA/Triton kernel nếu cần để aggregate dense \(H\) theo label pairs;
6. không materialize full one-hot \(P\) trừ khi batch nhỏ;
7. benchmark hai hướng:
   - dense \(H\)-based aggregation;
   - factor \(R^\top R\)-based computation nếu calibration factor có kích thước phù hợp.

**Lưu ý lý thuyết:** nếu không update codebook ở mọi MM step, monotonicity vẫn giữ cho từng assignment step; khi codebook update được gọi bằng exact solve thì bước đó cũng giảm objective. Tần suất gọi codebook update là engineering schedule, không thay đổi nghiệm của từng subproblem khi nó được thực hiện.

---

# 34. Outer handling của constraints và dual variables

Bài toán scalar quantization là discrete/non-convex.

Do đó:

- KKT/strong duality chỉ sạch cho continuous convex relaxation;
- với discrete codebook, Lagrangian là một relaxation;
- không được claim rằng dual search tìm global optimum của constrained quantization problem.

Implementation phải luôn giữ một feasible fallback.

Initialization:

\[
q_{\mathrm{best}}=q_0.
\]

Vì:

\[
q_0
\]

định nghĩa chính budgets, nó luôn feasible.

Sau mỗi fixed-dual inner solve, tính:

\[
C_{\mathrm{KL}}
=
\frac12
e^\top
H_{\mathrm{KL}}
e,
\]

\[
C_w
=
\frac12
\|e\|_2^2.
\]

Nếu:

\[
C_{\mathrm{KL}}
\le
\epsilon_{\mathrm{KL}}
\]

và:

\[
C_w
\le
\epsilon_w,
\]

candidate là feasible.

Trong số các feasible candidates, giữ candidate có local NLL surrogate thấp nhất:

\[
\boxed{
g^\top e
+
\frac12
e^\top H_{\mathrm{NLL}}e.
}
\]

---

# 35. Recommended MVP dual controller

Phần fixed-dual MM + exact codebook là core solver có descent guarantee.

Outer dual controller có thể triển khai bằng normalized projected dual ascent.

Với một row:

\[
v_{\mathrm{KL}}
=
\frac{C_{\mathrm{KL}}}
{\epsilon_{\mathrm{KL}}}
-
1,
\]

\[
v_w
=
\frac{C_w}
{\epsilon_w}
-
1.
\]

Một scale-aware choice không cần manual loss-weight tuning:

\[
\bar h_N
=
\frac{\operatorname{tr}(H_{\mathrm{NLL}})}{n},
\]

\[
\bar h_K
=
\frac{\operatorname{tr}(H_{\mathrm{KL}})}{n}.
\]

Đặt numerical scales:

\[
\eta_\mu
=
\frac{\bar h_N}
{\bar h_K+\varepsilon_{\mathrm{num}}},
\]

\[
\eta_\nu
=
\bar h_N.
\]

Update:

\[
\boxed{
\mu
\leftarrow
\max
\left(
0,
\mu
+
\eta_\mu v_{\mathrm{KL}}
\right)
}
\]

\[
\boxed{
\nu
\leftarrow
\max
\left(
0,
\nu
+
\eta_\nu v_w
\right).
}
\]

Đây là một **implementation controller**, không phải phần của fixed-dual monotonicity theorem.

Recommended safety:

- luôn giữ `best_feasible`, initialized bằng \(q_0\);
- outer loop không được overwrite final output bằng candidate infeasible;
- log constraint ratios sau mỗi outer iteration;
- nếu discrete assignments oscillate, stop và trả `best_feasible`.

Sau khi có prototype ổn định, dual controller có thể được nghiên cứu/tối ưu riêng mà không thay đổi inner MM/codebook derivation.

---

# 36. End-to-end algorithm

## Inputs

- full-precision model;
- calibration tokens;
- bit-width \(b\);
- \(K=2^b\);
- number of curvature groups \(G\);
- `sqllm` initialization from GuidedQuant;
- number of KL Fisher probes \(M\).

## Offline/statistics stage

### A. Run `sqllm`

Obtain:

\[
q_0,
\]

initial labels/codebooks nếu implementation expose chúng.

### B. Collect NLL statistics

Using ground-truth next-token labels:

- signed weight gradient \(g\);
- layer inputs \(X\);
- NLL layer-output gradients;
- grouped \(H_{\mathrm{NLL}}\).

### C. Collect KL Fisher statistics

Using teacher output distribution:

- sample pseudo-labels;
- backward log-prob scores;
- collect layer-output score gradients;
- grouped \(H_{\mathrm{KL}}\).

### D. Precompute

For each curvature group:

\[
r_N[i]
=
\sum_j
|
(H_{\mathrm{NLL}})_{ij}
|,
\]

\[
r_K[i]
=
\sum_j
|
(H_{\mathrm{KL}})_{ij}
|.
\]

---

## Quantization stage per row

Initialize:

\[
q=q_0.
\]

\[
e_0=q_0-w.
\]

Budgets:

\[
\epsilon_{\mathrm{KL}}
=
\frac12e_0^\top H_{\mathrm{KL}}e_0.
\]

\[
\epsilon_w
=
\frac12\|e_0\|^2.
\]

Set:

```text
best_feasible = q0
```

Initialize duals.

Then outer loop:

1. build implicit

\[
A
=
H_{\mathrm{NLL}}
+
\mu H_{\mathrm{KL}}
+
\nu I;
\]

2. run fixed-dual alternating solver:
   - Parallel MM assignment;
   - exact codebook update;
   - repeat until inner convergence;

3. compute KL and weight constraint values;

4. if feasible:
   - compare NLL surrogate;
   - update `best_feasible`;

5. update \(\mu,\nu\);

6. warm-start next outer iteration from current candidate or best feasible state.

Output:

\[
\boxed{
q_{\mathrm{best\_feasible}}.
}
\]

---

# 37. Pseudocode

```text
INPUT:
    FP weights W
    calibration data
    bit-width b, K = 2^b
    number of curvature groups G
    GuidedQuant sqllm initializer
    KL Fisher probes M

STAGE 1: INITIALIZATION
    Q0, labels0, codebooks0 = run_sqllm(...)

STAGE 2: NLL STATISTICS
    g = grad(mean_ground_truth_NLL, W)
    collect X
    collect per-token d(NLL_t)/dZ_layer
    H_NLL[group] = GuidedQuantGroupedFisher(X, dNLL_dZ)

STAGE 3: KL STATISTICS
    for probe in 1..M:
        y_probe ~ teacher_output_distribution
        score = log p_teacher(y_probe)
        collect per-token d(score_t)/dZ_layer
        accumulate KL Fisher
    H_KL[group] = GuidedQuantGroupedFisher(X, score_grads)

STAGE 4: PRECOMPUTE
    r_NLL[group] = abs(H_NLL[group]).sum(dim=-1)
    r_KL[group]  = abs(H_KL[group]).sum(dim=-1)

FOR each curvature group:
    batch rows sharing H_NLL[group], H_KL[group]

    FOR each row:
        q = q0
        labels = labels0
        codebook = codebook0

        e0 = q0 - w
        eps_KL = 0.5 * e0 @ H_KL @ e0
        eps_w  = 0.5 * dot(e0, e0)

        best_feasible = q0
        best_nll_surrogate =
            g @ e0 + 0.5 * e0 @ H_NLL @ e0

        initialize mu, nu

    FOR outer_iter:
        FOR inner_iter:
            E = Q - W

            S =
                G
                + E @ H_NLL
                + mu[:, None] * (E @ H_KL)
                + nu[:, None] * E

            Mdiag =
                r_NLL[None, :]
                + mu[:, None] * r_KL[None, :]
                + nu[:, None]

            target = Q - S / Mdiag

            labels_new =
                batched_nearest_sorted_codeword(
                    target,
                    codebooks,
                    current_labels=labels
                )

            Q = gather(codebooks, labels_new)

            codebooks =
                exact_codebook_update(
                    W, G,
                    H_NLL, H_KL,
                    labels_new,
                    mu, nu
                )

            sort codebooks and remap labels
            Q = gather(codebooks, labels)

            if labels stable:
                break

        compute C_KL, C_w

        if feasible:
            compute local NLL surrogate
            update best_feasible if better

        update duals mu, nu

    write best_feasible back to model
```

---

# 38. Recommended implementation modules

```text
stats/
    collect_nll_gradient.py
    collect_nll_curvature.py
    collect_kl_fisher.py
    grouped_guidedquant_hessian.py

init/
    sqllm_adapter.py

solver/
    parallel_mm_assignment.py
    exact_codebook_update.py
    dual_controller.py

quant/
    method_a_quantizer.py

utils/
    batched_searchsorted.py
    curvature_io.py
    objective_metrics.py
```

---

# 39. Suggested tensor interfaces

## Per module

```text
weight:
    [d_out, d_in]

g:
    [d_out, d_in]

group_id:
    [d_out]

H_nll:
    [num_curvature_groups, d_in, d_in]

H_kl:
    [num_curvature_groups, d_in, d_in]

row_abs_nll:
    [num_curvature_groups, d_in]

row_abs_kl:
    [num_curvature_groups, d_in]

labels:
    [d_out, d_in]

codebooks:
    [d_out, K]

q:
    [d_out, d_in]

mu:
    [d_out]

nu:
    [d_out]

eps_kl:
    [d_out]

eps_w:
    [d_out]
```

---

# 40. Numerical precision

Recommended:

- original weights: model dtype;
- \(g\): FP32 accumulation;
- Hessian accumulation: FP32;
- MM GEMM: TF32/FP32 or validated mixed precision;
- codebook \(K\times K\) solve: FP32;
- objective/constraint checks: FP64 if debugging, FP32 production.

Do not silently downcast Hessian statistics to FP16 before validating stability.

---

# 41. Debug metrics cần log

## Statistics

```text
max_abs_g
mean_abs_g
median_abs_g

trace_H_nll
trace_H_kl

min_diag_H_nll
median_diag_H_nll
max_diag_H_nll

min_diag_H_kl
median_diag_H_kl
max_diag_H_kl

min_row_abs_nll
max_row_abs_nll

min_row_abs_kl
max_row_abs_kl
```

## Initialization

```text
max_abs_w
max_abs_q0
max_abs_e0

eps_kl
eps_w

q0_local_nll_surrogate
```

## MM

```text
fixed_dual_objective_before
fixed_dual_objective_after_assignment
fixed_dual_objective_after_codebook

num_label_changes
max_abs_target
max_abs_codebook
```

Expected invariant for fixed duals:

```text
objective_after_assignment <= objective_before
objective_after_codebook   <= objective_after_assignment
```

## Constraints

```text
C_KL / eps_KL
C_w  / eps_w
```

Final output must satisfy:

```text
C_KL <= eps_KL
C_w  <= eps_w
```

up to numerical tolerance.

---

# 42. Unit tests

## Test 1: Hessian PSD

For random vectors \(v\):

\[
v^\top H_{\mathrm{NLL}}v\ge0
\]

và:

\[
v^\top H_{\mathrm{KL}}v\ge0.
\]

Cho phép numerical tolerance nhỏ.

---

## Test 2: Majorizer

Random \(v\):

\[
v^\top Mv
\ge
v^\top Av.
\]

---

## Test 3: MM assignment monotonicity

Với fixed:

\[
w,g,H_N,H_K,\mu,\nu,\mathcal C,
\]

one parallel assignment step phải thỏa:

\[
J_{\mathrm{new}}
\le
J_{\mathrm{old}}.
\]

---

## Test 4: Exact codebook update

Sau codebook solve:

\[
J(c_{\mathrm{new}})
\le
J(c_{\mathrm{old}}).
\]

Với nonsingular active system:

\[
\|
Lc-b
\|
\]

phải rất nhỏ.

---

## Test 5: Source correctness

Hai collector phải được test riêng.

NLL collector:

```text
uses ground-truth labels
```

KL collector:

```text
uses pseudo-label sampled from teacher distribution
does not use ground-truth labels
```

---

## Test 6: Reduction normalization

So sánh curvature thu bằng:

- per-token/sum-loss implementation;
- mean-loss implementation đã undo scaling.

Hai kết quả phải khớp về scale.

---

## Test 7: q0 feasibility

Theo construction:

\[
C_{\mathrm{KL}}(q_0)
=
\epsilon_{\mathrm{KL}}
\]

và:

\[
C_w(q_0)
=
\epsilon_w.
\]

---

# 43. Những lỗi không được lặp lại

## 43.1. Không dùng:

\[
q_0
=
w-H^{-1}g.
\]

Đây là nguyên nhân có thể tạo giant shift ở flat directions.

---

## 43.2. Không lấy \(g/H\) trực tiếp làm initialization.

---

## 43.3. Không gọi NLL empirical Fisher là exact KL Hessian.

---

## 43.4. Không dùng ground-truth NLL activation gradient để xây \(H_{\mathrm{KL}}\).

---

## 43.5. Không dùng teacher-score gradient để thay signed NLL gradient \(g\).

---

## 43.6. Không square mean gradient rồi coi là mean Fisher nếu reduction scaling không đúng.

---

## 43.7. Không dùng LNQ cyclic coordinate descent.

Assignment của Method A là synchronous Parallel MM.

---

## 43.8. Không materialize \([N,K]\) hoặc \([B,n,K]\) candidate tensors khi \(K\) lớn nếu có thể dùng sorted codebook + `searchsorted`.

---

# 44. Core contribution summary

Method A có bốn phần chính.

## 1. Constrained end-loss formulation

\[
\min
L_{\mathrm{NLL}}(q)
\]

subject to:

\[
D_{\mathrm{KL}}(p_w\|p_q)
\le
\epsilon_{\mathrm{KL}}
\]

và:

\[
\frac12\|q-w\|^2
\le
\epsilon_w.
\]

Budgets tự động lấy từ `sqllm` reference quantization.

---

## 2. Hai curvature tách biệt

\[
H_{\mathrm{NLL}}
\]

từ ground-truth NLL.

\[
H_{\mathrm{KL}}
\]

từ teacher output score Fisher.

Cả hai dùng GuidedQuant-style dense grouped approximation:

\[
X^\top
\operatorname{Diag}(s)
X.
\]

---

## 3. Parallel MM assignment

Fixed-dual matrix:

\[
A
=
H_{\mathrm{NLL}}
+
\mu H_{\mathrm{KL}}
+
\nu I.
\]

Gradient:

\[
s
=
g+Ae.
\]

Diagonal majorizer:

\[
m_i
=
r_i^{\mathrm{NLL}}
+
\mu r_i^{\mathrm{KL}}
+
\nu.
\]

Target:

\[
t_i
=
q_i
-
\frac{s_i}{m_i}.
\]

Parallel projection:

\[
q_i^+
=
\operatorname{NearestCodeword}(t_i).
\]

---

## 4. Exact codebook update

\[
\boxed{
(P^\top AP)c
=
P^\top Aw
-
P^\top g.
}
\]

Solve hệ nhỏ \(K\times K\).

---

# 45. Final implementation flow

\[
\boxed{
\text{GuidedQuant sqllm init}
}
\]

\[
\downarrow
\]

\[
\boxed{
q_0,\;
\epsilon_{\mathrm{KL}},\;
\epsilon_w
}
\]

\[
\downarrow
\]

\[
\boxed{
g
\text{ from ground-truth NLL}
}
\]

\[
+
\]

\[
\boxed{
H_{\mathrm{NLL}}
\text{ from GuidedQuant-style NLL empirical Fisher}
}
\]

\[
+
\]

\[
\boxed{
H_{\mathrm{KL}}
\text{ from GuidedQuant-style teacher-score Fisher}
}
\]

\[
\downarrow
\]

\[
\boxed{
A
=
H_{\mathrm{NLL}}
+
\mu H_{\mathrm{KL}}
+
\nu I
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{Parallel MM Assignment}
\leftrightarrow
\text{Exact Codebook Update}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{keep best feasible quantized solution}
}
\]

---

# 46. Theoretical claims that are safe to make

Có thể claim:

1. Với fixed \(\mu,\nu\), GuidedQuant-style Fisher approximations là PSD.
2. Absolute-row-sum diagonal matrix majorizes each dense curvature matrix.
3. Parallel MM assignment monotonically decreases fixed-dual objective.
4. Exact codebook update monotonically decreases fixed-dual objective.
5. Alternating assignment/codebook updates monotonically decrease fixed-dual objective.

Không nên claim:

1. global optimum của discrete constrained problem;
2. strong duality cho scalar quantization;
3. outer dual updates monotonically decrease cùng một objective;
4. \(H_{\mathrm{NLL}}\) empirical Fisher là exact NLL Hessian;
5. grouped \(H_{\mathrm{KL}}\) là full exact model Fisher.

---

# 47. Implementation priority

Thứ tự code/test nên là:

### Phase 1
Reuse `sqllm` để tạo \(q_0\).

### Phase 2
Collect signed NLL gradient \(g\).

### Phase 3
Collect GuidedQuant-style \(H_{\mathrm{NLL}}\).

### Phase 4
Collect teacher-score Fisher \(H_{\mathrm{KL}}\).

### Phase 5
Verify scale và PSD của hai Hessian.

### Phase 6
Implement fixed-dual Parallel MM assignment.

### Phase 7
Verify monotonicity assignment.

### Phase 8
Implement exact \(K\times K\) codebook update.

### Phase 9
Verify monotonicity alternating inner loop.

### Phase 10
Add budgets từ \(q_0\) và outer dual controller.

### Phase 11
Batch rows sharing curvature và optimize GPU execution.

---

# 48. One-line formula sheet

Original local constrained problem:

\[
\boxed{
\min_{q\in\mathcal Q_K}
g^\top(q-w)
+
\frac12(q-w)^\top
H_{\mathrm{NLL}}
(q-w)
}
\]

subject to:

\[
\boxed{
\frac12(q-w)^\top
H_{\mathrm{KL}}
(q-w)
\le
\epsilon_{\mathrm{KL}}
}
\]

\[
\boxed{
\frac12\|q-w\|^2
\le
\epsilon_w
}
\]

with:

\[
\epsilon_{\mathrm{KL}}
=
\frac12(q_0-w)^\top
H_{\mathrm{KL}}
(q_0-w),
\]

\[
\epsilon_w
=
\frac12
\|q_0-w\|^2,
\]

\[
q_0
=
Q_{\mathrm{sqllm}}(w).
\]

Fixed dual:

\[
\boxed{
A
=
H_{\mathrm{NLL}}
+
\mu H_{\mathrm{KL}}
+
\nu I.
}
\]

MM gradient:

\[
\boxed{
s
=
g+A(q-w).
}
\]

Majorizer:

\[
\boxed{
m
=
r_{\mathrm{NLL}}
+
\mu r_{\mathrm{KL}}
+
\nu.
}
\]

MM target:

\[
\boxed{
t
=
q-s\oslash m.
}
\]

Assignment:

\[
\boxed{
q_i^+
=
\operatorname{NearestCodeword}(t_i).
}
\]

Exact codebook:

\[
\boxed{
(P^\top AP)c
=
P^\top Aw-P^\top g.
}
\]

---

# 49. Reference links

GuidedQuant GitHub:

https://github.com/snu-mllab/GuidedQuant

GuidedQuant OpenReview:

https://openreview.net/forum?id=ZawsPjlIGu

GuidedQuant paper/PMLR:

https://proceedings.mlr.press/v267/kim25d.html

The implementation should reuse or adapt GuidedQuant's gradient/Hessian collection infrastructure where useful, while preserving the loss-source separation defined in this document.

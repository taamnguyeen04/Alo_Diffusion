$$Y = W * X + b = W * [x, e] + b$$ $$Y = \underbrace{(W_x * x)}{\text{Đặc trưng ảnh}} + \underbrace{(W_e * e)}{\text{Điều kiện cảm xúc}} + b$$



$$h_{mới}[b, c, x, y] = \underbrace{\gamma[b, c]}{\text{Hệ số kênh c}} \times h{cũ}[b, c, x, y] + \underbrace{\beta[b, c]}_{\text{Độ lệch kênh c}}$$




---
$$\mathcal{L}{wav} = \lambda{ll} \cdot \mathcal{L}{VGG}(LL{pred}, LL_{target}) + \lambda_{hi} \cdot \mathcal{L}_{HF}$$

LL Band (VGG Perceptual): $$\mathcal{L}{VGG} = \sum{l \in {relu1_2, relu2_2, relu3_3}} | \phi_l(LL_{pred}) - \phi_l(LL_{target}) |_1$$

HF Bands (L1 on 6 orientations): $$\mathcal{L}{HF} = \frac{1}{2} \left( | HF^{real}{pred} - HF^{real}{target} |1 + | HF^{imag}{pred} - HF^{imag}{target} |_1 \right)$$

$$\Delta\text{Final} = \Delta\text{UNet_Local} + (\Delta\text{Modulator_Global} \times \text{Direction_Attention})$$


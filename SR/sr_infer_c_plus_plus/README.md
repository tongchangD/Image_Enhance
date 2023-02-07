## 超分C++推理代码

简陋的图像超分推理代码，采用jit的形式推理

代码利用libtorch

方式：

1、第一步：到处 pt模型

```python
sf = 4
model_path = "./model_zoo/models.pth"
img_path = "/home/tcd/Desktop/杂图/ADE_val_00000114.jpg"
filename = os.path.split(img_path)[-1]
LR = util.imread_uint(img_path, n_channels=3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = util.uint2tensor4(LR).to(device)
model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)
model.load_state_dict(torch.load(model_path), strict=True)
model.to(device)
model.eval()
model_dada = torch.jit.trace(model.forward,LR)
model_dada.save("/media/tcd/data/C++/SR/model/model.pt")
model_tcd = torch.jit.load('/media/tcd/data/C++/SR/model/model.pt')
SR = model_tcd.forward(LR)
print("output_tensor ", SR.shape)
```

2、将生成模型导入到/model/下

3、编译运行即可使用，需要更改文件名。

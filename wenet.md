

> https://github.com/wenet-e2e/wenet

## 第1节: 端到端语音识别基础

语音识别任务中，输入是语音，输出是文本，两个序列不等长，但是时序上有同步性。传统语音识别通过HMM来建模输出和输入不等长和时序同步性，并对音素，词典，语言模型分层次建模。

在传统的HMM框架里，声学模型往往在三音素的hmm状态级别建模，与语言模型分开建模训练，在训练时，任务的整体目标被割裂成多个训练目标。另外，传统的HMM框架框架的模型训练会涉及到很多过程，包括上下文相关音素，音素聚类，HMM-GMM训练，强制对齐等，非常繁琐。

下面是一个HMM-DNN模型的训练过程，仅用于说明训练过程的复杂：

- 对于每个句子扩展为单音素序列，用前向后向EM训练，得到单音素的hmm-单高斯model1。
- 用model1对句子做对齐，单高斯进行2倍分裂，更新模型，迭代这个对齐/分裂的过程n次，得到单音素的hmm-gmm模型model2.
- 用model2对句子做对齐，将音素根据上下文扩展为三音素，使用单高斯学习每个音素的决策树，，最后每个叶子结点对应一个单高斯. 得到一个三音素-hmm-单高斯模型model3
- 类似于第2步，用model3不停迭代分裂高斯，得到三音素hmm-gmm的model4
- model4对句子做对齐，对齐数据用于帧级别NN训练.
- 不断迭代

一般在传统HMM框架下，会先利用HMM-GMM模型，通过对齐的方式，得到帧级别的对应标注，再通过帧级别损失函数来优化神经网络模型， 不过这也不是必须的，HMM框架也可以不做帧对齐，比如论文End-to-end speech recognition using lattice-free MMI中直接进行训练。

近几年，基于神经网络的端到端建模方式则更佳简洁

- 直接以目标单元作为建模对象，比如中文使用`字`，英文使用`字符`或者`BPE`.
- 通过特殊的模型（目标函数），处理输入输出对齐未知的问题。

### CTC目标函数

传统语音识别通过HMM来约束输出和输入的对齐方式（时间上保持单调），CTC是一种特殊的HMM约束。

CTC本质上对所有合法的输出和输入对齐方式进行了穷举，所谓合法，即对齐后的输出序列能够按CTC规则规约得到的原标注序列，则为合法对齐。

使用CTC目标函数会引入一个blank的输出单元，CTC规约规则为：

- 连续的相同字符进行合并
- 移除blank字符

一个例子：

某段语音数据，输入帧数为7帧（此处仅用于举例），原始的标注序列为“出门问问”，为4个字，但是网络需要输出7个单元，输出和输入一一对应。CTC模型中，通过对原标注内的4个单元进行重复和插入blank来扩展为7个单元，下面两种都是可能的扩展序列，其中`-`为blank，如果扩展序列通过CTC规则规约可以得到原标注序列，则改扩展序列称为合法对齐序列。

```
出-门问问-问  -> 出门问问
出-门--问问 -> 出门问
```

第一个对齐序列`出-门问问-问`是合法对齐序列，第二个对齐序列`出-门--问问`不是合法对齐序列。

除了`出-门问问-问`还有很多其他合法序列，比如

```
出出门问问-问
出出出门问-问
出-门-问-问
出--门问-问
...
```

CTC目标函数的思想是: 既然不知道哪种对齐关系是正确的，那就最大化所有合法CTC对齐的概率之和。所以对于这个样本，目标就是最大化如下概率。

```
P(出门问问|X) = P(出-门问问-问|X) + P(出出门问问-问|X)
              + P(出出出门问-问|X)+ ... + P(出--门问-问|X)
```

求这个目标函数梯度的一种方式是穷举所有的有效CTC对齐，分别求梯度相加。但是这种方法复杂度太高。由于CTC本身结构特点，存在一种更高效的动态规划算法，可以极大的提升速度。具体可参考论文 CTC-paper 和文章Eesen中的CTC实现。

解码时，模型对每一个输入帧都给出输出，这种解码方法称为`Frame同步`解码。若某些帧输出为blank或者和前一帧是重复的字符，则可以合并。由于穷举序列中blank占的个数最多。最后模型倾向于输出尽量少的非blank字符，因此解码序列中往往每个非blank字符只输出一次，这个叫做CTC的**尖峰效应。**

### Attention-based Encoder Decoder

Attention-based Encoder Decoder简称AED，也叫Seq2Seq框架，在ASR领域里，该框架也叫做LAS（Listen, Attend and Spell）。

模型中的encoder对输入序列（语音）进行信息提取，decoder则是一个在目标序列（文本）上的自回归模型（输入之前的单元，预测下一个单元），同时在自回归计算时，通过attention方式去获取encoder的输出编码信息，从而能够利用到输入序列的信息。

这种建模方式，可以不必显示建模输出和输入之间的对齐关系，而是利用attention机制交给网络去学习出**隐含的对齐**。相比如CTC，AED允许输入输出单元之间存在时序上的交换，因此特别适用于机器翻译这种任务。但是对于语音识别或者语音合成这些存在时序单调性的任务，这种无约束反而会带来一些问题。

**多说一句**

CTC没有显示构建文本之间的关系，RNN-t模型是一种显示建模了文本关系的帧同步解码的模型。

标准的AED中，decoder和encoder之间cross-attention需要看到encoder的完整序列，所以无法进行流式识别。

可利用GMM-attention/Mocha/MMA等单调递进的局部Attention方法进行改进。

### 联合建模

研究者发现，联合使用CTC loss和AED可以有效的加速训练收敛，同时得到更好的识别结果。目前这个方法已经成为端到端学习的标准方案。

在解码时，同时使用CTC和AED的输出，可以提高识别率，但是由于AED本身是非流式的解码，在Wenet中，则没采用联合解码的方式，而是采用了先使用CTC解码，再用AED对CTC的Nbest结果进行Rescoring，这样即结合了两种模型的效果，又可以应用于流式场景。

### 神经网络类型

常用的神经网络类型包括DNN，CNN，RNN，Self-attention等，这些方法进行组合，衍生除了各种模型，Wenet中，对于encoder网络部分，支持Transformer和Conformer两种网络。decoder网络部分，支持Transformer网络。

Transformer由多个Transformer Block堆叠，每个Block中会使用self-attention，res，relu，ff层。

Conformer由多个Conformer Block堆叠，每个Block中会使用conv，self-attention，res，relu，ff层。

**降采样/降帧率**

输入序列越长，即帧的个数越多，网络计算量就越大。而在语音识别中，一定时间范围内的语音信号是相似的，多个连续帧对应的是同一个发音，另外，端到端语音识别使用建模单元一般是一个时间延续较长的单元（粗粒度），比如建模单元是一个中文汉字，假如一个汉字用时0.2s，0.2s对应20帧，如果将20帧的信息进行合并，比如合并为5帧，则可以线性的减少后续encoder网络的前向计算、CTC loss和AED计算cross attention时的开销。

可以用不同的神经网络来进行降采样，Wenet中使用的是2D-CNN。

### 流式语音识别

虽然CTC解码是`Frame同步`的，但是要想支持低延迟的流式识别，Encoder中的计算对右侧的依赖不能太长。标准的Fully self-attention会对依赖整个序列，不能进行流式计算，因此wenet采用了基于chunk的attention，将序列划分为多个固定大小的chunk，每个chunk内部的帧不会依赖于chunk右侧的帧。同时，连续堆叠的convolution层会带来较大的右侧依赖，wenet则采用了因果卷积来避免convolution层的右侧依赖。

## 第2节: Wenet中的神经网络设计与实现

Wenet的代码借鉴了Espnet等开源实现，比较简洁，但是为了实现基于chunk的流式解码，以及处理batch内不等长序列，引入的一些实现技巧，比如cache和mask，使得多处的代码在初次阅读时不易理解，可在第一步学习代码时略过相关内容。

### 模型入口ASEModel

核心模型的代码位于`wenet/transformer/`目录

模型定义：ASRModel的init中定义了encoder, decoder, ctc, criterion_att几个基本模块。其整体网络拓扑如下图所示。

- encoder是Shared Encoder，其中也包括了Subsampling网络。
- decoder是Attention-based Decoder网络
- ctc是ctc Decoder网络（很简单，仅仅是前向网络和softmax）和ctc loss
- criterion_att是attention-based decoder的自回归似然loss，实际是一个LabelSmoothing的loss。

![图片](https://mmbiz.qpic.cn/mmbiz_png/FNwn7wEvTjhkdMVMnMTVg2uibT3TsOAYjqM8kL4dktTav5WG9FYpGhUyYzgc2mN3rCBafP0LicMjyrWFuR9ian7GA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### 创建模型

```
def init_asr_model(config):
```

该方法根据传入的config，创建一个ASRModel实例。config内容由训练模型时使用yaml文件提供。这个创建仅仅是构建了一个初始化模型，其参数是随机的，可以通过model.load_state_dict(checkpoint)从训练好的模型中加载参数。

### 前向计算

pytorch框架下，只需定义模型的前向计算forword,。对于每个Module，可以通过阅读forward代码来学习其具体实现。

### 其他接口

ASRModel除了定义模型结构和实现前向计算用于训练外，还有两个功能：

- 提供多种python的解码接口

```
recognize() # attention decoder
attention_rescoring() # CTC + attention rescoring
ctc_prefix_beam_search() # CTC prefix beamsearch
ctc_greedy_search() # CTC greedy search
```

- 提供runtime中需要使用的接口。 这些接口均有@torch.jit.export注解，可以在C++中调用

```
subsampling_rate()
right_context()
sos_symbol()
eos_symbol()
forward_encoder_chunk()
forward_attention_decoder()
ctc_activation()
```

其中比较重要的是：

- `forward_attention_decoder()` Attention Decoder的序列forward计算，非自回归模式。
- `ctc_activation()` CTC Decoder forward计算
- `forward_encoder_chunk()` 基于chunk的Encoder forward计算

### Encoder网络

**wenet/transformer/encoder.py**

Wenet的encoder支持Transformer和Conformer两种网络结构，实现时使用了模版方法的设计模式进代码复用。BaseEncoder中定义了如下统一的前向过程，由TransformerEncoder，ConformerEncoder继承BaseEncoder后分别定义各自的self.encoders的结构。

```
class BaseEncoder(torch.nn.Module):
  def forward(...):
      xs, pos_emb, masks = self.embed(xs, masks)
      chunk_masks = add_optional_chunk_mask(xs, ..)
      for layer in self.encoders:
          xs, chunk_masks, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
      if self.normalize_before:
          xs = self.after_norm(xs)
```

可以看到Encoder分为两大部分

- self.embed是Subsampling网络
- self.encoders是一组相同结构网络（Encoder Blocks）的堆叠

除了forward，Encoder还实现了两个方法，此处不展开介绍。

- `forward_chunk_by_chunk`，python解码时，模拟流式解码模式基于chunk的前向计算。
- `forward_chunk`, 单次基于chunk的前向计算，通过ASRModel导出为`forward_encoder_chunk()`供runtime解码使用。

### Subsampling网络

**wenet/transformer/subsampling.py**

语音任务里有两种使用CNN的方式，一种是2D-Conv，一种是1D-Conv：

- 2D-Conv: 输入数据看作是深度(通道数）为1，高度为F（Fbank特征维度，idim），宽度为T（帧数）的一张图.
- 1D-Conv: 输入数据看作是深度(通道数）为F（Fbank特征维度)，高度为1，宽度为T（帧数）的一张图.

Kaldi中著名的TDNN就是是1D-Conv，在Wenet中采用2D-Conv来实现降采样。

Wenet中提供了多个降采样的网络，这里选择把帧率降低4倍的网络`Conv2dSubsampling4`来说明。

```
class Conv2dSubsampling4(BaseSubsampling):
    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim)
        self.pos_enc = pos_enc_class
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) / 2 * stride  * frame_rate_of_this_layer
        self.subsampling_rate = 4
        # 6 = (3 - 1) / 2 * 2 * 1 + (3 - 1) / 2 * 2 * 2
        self.right_context = 6
```

我们可以把一个语音帧序列[T,D]看作宽是T，高是D，深度为1的图像。`Conv2dSubsampling4`通过两个`stride=2`的2d-CNN，把图像的宽和高都降为1/4. 因为图像的宽即是帧数，所以帧数变为1/4.

```
torch.nn.Conv2d(1, odim, kernel_size=3, stride=2)
torch.nn.Conv2d(odim, odim, kernel_size=3, stride=2)
```

**具体的实现过程**

```
def forward(...):
    x = x.unsqueeze(1)  # (b, c=1, t, f)
    x = self.conv(x)
    b, c, t, f = x.size()
    x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
    x, pos_emb = self.pos_enc(x, offset)
    return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2]
```

- x = x.unsqueeze(1) # (b, c=1, t, f) 增加channel维，以符合2dConv需要的数据格式。
- conv(x)中进行两次卷积，此时t维度约等于原来的1/4，因为没加padding，实际上是从长度T变为长度((T-1)/2-1)/2）。注意经过卷积后深度不再是1。
- view(b, t, c * f) 将深度和高度合并平铺到同一维，然后通过self.out(）对每帧做Affine变换.
- pos_enc(x, offset) 经过subsampling之后，帧数变少了，此时再计算Positional Eembedding。

在纯self-attention层构建的网络里，为了保证序列的顺序不可变性而引入了PE，从而交换序列中的两帧，输出会不同。但是由于subsampling的存在，序列本身已经失去了交换不变性，所以其实PE可以省去。

x_mask是原始帧率下的记录batch各序列长度的mask，在计算attention以及ctc loss时均要使用，现在帧数降低了，x_mask也要跟着变化。

返回独立的pos_emb，是因为在relative position attention中，需要获取relative pos_emb的信息。在标准attention中该返回值不会被用到。

![图片](https://mmbiz.qpic.cn/mmbiz_png/FNwn7wEvTjhkdMVMnMTVg2uibT3TsOAYj0arNOztdOX6IuBIHUOZKbia85c3SQIQQnW2yFnl65h49Nnw3bSDu5nA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**上下文依赖**

注意Conv2dSubsampling4中的这两个变量。

```
self.subsampling_rate = 4
self.right_context = 6
```

这两个变量都在asr_model中进行了导出，在runtime时被使用，他们的含义是什么？

在CTC或者WFST解码时，我们是一帧一帧解码，这里的帧指的是subsample之后的帧。我们称为解码帧，而模型输入的帧序列里的帧（subsample之前的）称为原始语音帧。

在图里可以看到

- 第1个解码帧，需要依赖第1到第7个原始语音帧。
- 第2个解码帧，需要依赖第5到第11个原始语音帧。

subsampling_rate: 对于相邻两个解码帧，在原始帧上的间隔。

right_context: 对于某个解码帧，其对应的第一个原始帧的右侧还需要额外依赖多少帧，才能获得这个解码帧的全部信息。

在runtime decoder中，每次会送入一组帧进行前向计算并解码，一组（chunk）帧是定义在解码帧级别的，在处理第一个chunk时，接受输入获得当前chunk需要的所有的context，之后每次根据chunk大小和subsampling_rate获取新需要的原始帧。比如，chunk_size=1，则第一个chunk需要1-7帧，第二个chunk只要新拿到8-11帧即可。

```
# runtime/core/decoder/torch_asr_decoder.cc
TorchAsrDecoder::AdvanceDecoding()
    if (!start_) {                      // First chunk
      int context = right_context + 1;  // Add current frame
      num_requried_frames = (opts_.chunk_size - 1) * subsampling_rate + context;
    } else {
      num_requried_frames = opts_.chunk_size * subsampling_rate;
    }
```

### Encoder Block

对于Encoder, Wenet提供了Transformer和Conformer两种结构，Conformer在Transformer里引入了卷积层，是目前语音识别任务效果最好的模型之一。强烈建议阅读这篇文章 The Annotated Transformer, 了解Transformer的结构和实现。

Transformer的self.encoders由一组TransformerEncoderLayer组成

```
self.encoders = torch.nn.ModuleList([
    TransformerEncoderLayer(
        output_size,
        MultiHeadedAttention(attention_heads, output_size,
                              attention_dropout_rate),
        PositionwiseFeedForward(output_size, linear_units,
                                dropout_rate), dropout_rate,
        normalize_before, concat_after) for _ in range(num_blocks)
])
```

Conformer的self.encoders由一组ConformerEncoderLayer组成

```
self.encoders = torch.nn.ModuleList([
    ConformerEncoderLayer(
        output_size,
        RelPositionMultiHeadedAttention(*encoder_selfattn_layer_args),
        PositionwiseFeedForward(*positionwise_layer_args),
        PositionwiseFeedForward(*positionwise_layer_args)
        if macaron_style else None,
        ConvolutionModule(*convolution_layer_args)
        if use_cnn_module else None,
        dropout_rate,
        normalize_before,
        concat_after,
    ) for _ in range(num_blocks)
])
```

仅介绍ConformerEncoderLayer，其涉及的主要模块有：

- RelPositionMultiHeadedAttention
- PositionwiseFeedForward
- ConvolutionModule

如果不考虑cache，使用normalize_before=True，feed_forward_macaron=True，则wenet中的ConformerEncoderLayer的forward可以简化为

```
class ConformerEncoderLayer(nn.Module):
    def forward(...):
        residual = x
        x = self.norm_ff_macaron(x)
        x = self.feed_forward_macaron(x)
        x = residual + 0.5 * self.dropout(x)

        residual = x
        x = self.norm_mha(x)
        x_att = self.self_attn(x, x, x, pos_emb, mask)
        x = residual + self.dropout(x_att)

        residual = x
        x = self.norm_conv(x)
        x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
        x = x + self.dropout(x)

        residual = x
        x = self.norm_ff(x)
        x = self.feed_forward(x)
        x = residual + 0.5 * self.dropout(x)

        x = self.norm_final(x)
```

可以看到，对于RelPositionMultiHeadedAttention，ConvolutionModule，PositionwiseFeedForward，都是前有Layernorm，后有Dropout，再搭配Residual。

**Conformer Block - RelPositionMultiHeadedAttention**

**wenet/transformer/attention.py**

attention.py中提供了两种attention的实现，`MultiHeadedAttention`和`RelPositionMultiHeadedAttention`。

`MultiHeadedAttention`用于Transformer。`RelPositionMultiHeadedAttention`用于Conformer。

原始的Conformer论文中提到的self-attention是Relative Position Multi Headed Attention，这是transformer-xl中提出的一种改进attention，和标准attention的区别在于，其中显示利用了相对位置信息，具体原理和实现可参考文章。Conformer ASR中的Relative Positional Embedding

注意，wenet中实现的Relative Position Multi Headed Attention是存在问题的, 但是由于采用正确的实现并没有什么提升，就没有更新成transformer-xl中实现。

**Conformer Block - PositionwiseFeedForward**

**wenet/transformer/positionwise_feed_forward.py**

PositionwiseFeedForward，对各个帧时刻输入均使用同一个矩阵权重去做前向Affine计算，即通过一个[H1, H2]的的前向矩阵，把[B, T, H1]变为[B，T，H2]。

**Conformer Block - ConvolutionModule**

**wenet/transformer/convolution.py**

ConvolutionModule结构如下

Wenet中使用了因果卷积(Causal Convolution)，即不看右侧上下文，这样无论模型含有多少卷积层，对右侧的上下文都无依赖。

原始的对称卷积，如果不进行左右padding，则做完卷积后长度会减小。

![图片](https://mmbiz.qpic.cn/mmbiz_png/FNwn7wEvTjhkdMVMnMTVg2uibT3TsOAYjg98lIM2b7ExGuAHY4NQwIuN2c7pwGUia1DMEVL5F4vMZkCxstrv71GQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

因此标准的卷积，为了保证卷积后序列长度一致，需要在左右各pad长度为(kernel_size - 1) // 2的 0.

因此标准的卷积，为了保证卷积后序列长度一致，需要在左右各pad长度为(kernel_size - 1) // 2的 0.

```
  if causal: # 使用因果卷积
    padding = 0 # Conv1D函数设置的padding长度
    self.lorder = kernel_size - 1 # 因果卷积左侧手动padding的长度
  else: # 使用标准卷积
    # kernel_size should be an odd number for none causal convolution
    assert (kernel_size - 1) % 2 == 0
    padding = (kernel_size - 1) // 2 # Conv1D函数设置的padding长度
    self.lorder = 0
```

因果卷积的实现其实很简单，只在左侧pad长度为kernel_size - 1的0，即可实现。如图所示。

```
if self.lorder > 0:
  if cache is None:
    x = nn.functional.pad(x, (self.lorder, 0), 'constant', 0.0)
```

### Attention based Decoder网络

对于Attention based Decoder, Wenet提供了自回归Transformer和双向自回归Transformer结构。所谓自回归，既上一时刻的网络输出要作为网络当前时刻的输入，产生当前时刻的输出。

在ASR整个任务中，Attention based Decoder的输入是当前已经产生的文本，输出接下来要产生的文本，因此这个模型建模了语言模型的信息。

这种网络在解码时，只能依次产生输出，而不能一次产生整个输出序列。

和Encoder中的attention层区别在于，Decoder网络里每层DecoderLayer，除了进行self attention操作(self.self_attn)，也和encoder的输出进行cross attention操作(self.src_attn)

另外在实现上，由于自回归和cross attention，mask的使用也和encoder有所区别。

### CTC Loss

**wenet/transformer/ctc.py**

CTC Loss包含了CTC decoder和CTC loss两部分，CTC decoder仅仅对Encoder做了一次前向线性计算，然后计算softmax.

```
    # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
    ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
    # ys_hat: (B, L, D) -> (L, B, D)
    ys_hat = ys_hat.transpose(0, 1)
    ys_hat = ys_hat.log_softmax(2)
    loss = self.ctc_loss(ys_hat, ys_pad, hlens, ys_lens)
    # Batch-size average
    loss = loss / ys_hat.size(1)
    return loss
```

CTC loss的部分则直接使用的torch提供的函数 `torch.nn.CTCLoss`.

```
    self.ctc_loss = torch.nn.CTCLoss(reduction=reduction_type)
```

### Attention based Decoder Loss

**wenet/transformer/label_smoothing_loss.py**

Attention-based Decoder的Loss是在最大化自回归的概率，在每个位置计算模型输出概率和样本标注概率的Cross Entropy。这个过程采用teacher forcing的方式，而不采用scheduled sampling。

每个位置上，样本标注概率是一个one-hot的表示，既真实的标注概率为1，其他概率为0. Smoothing Loss中，对于样本标注概率，将真实的标注概率设置为1-e，其他概率设为e/(V-1)。

### 网络的完整结构

通过`print()`打印出的`ASRModel`的网络结构。
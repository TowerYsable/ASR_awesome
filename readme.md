

[zzw922cn/awesome-speech-recognition-speech-synthesis-papers: Automatic Speech Recognition (ASR), Speaker Verification, Speech Synthesis, Text-to-Speech (TTS), Language Modelling, Singing Voice Synthesis (SVS), Voice Conversion (VC) (github.com)](https://github.com/zzw922cn/awesome-speech-recognition-speech-synthesis-papers#Automatic-Speech-Recognition)

[The Top 98 Automatic Speech Recognition Open Source Projects on Github (awesomeopensource.com)](https://awesomeopensource.com/projects/automatic-speech-recognition)

1 Deep Context 1808.02480.pdf

## Awesome Contextualization of E2E ASR

问题提出：对于产品级的自动语音识别（Automatic Speech Recognition, ASR）,能够适应专有领域的语境偏移（contextual bias），是一个很重要的功能。举个例子，对于手机上的ASR，系统要能准确识别出用户说的app的名字，联系人的名字等等，而不是发音相同的其他词。更具体一点，比如读作“Yao Ming”的这个词语，在体育领域可能是我们家喻户晓的运动员“姚明”，但是在手机上，它可能是我们通讯录里面一个叫做“姚敏”的朋友。如何随着应用领域的变化，解决这种偏差问题就是我们这个系列的文章要探索的主要问题。

对于传统的ASR系统，它们往往有独立的声学模型（AM）、发音词典（PM）、以及语言模型（LM），当需要对特定领域进行偏移时，可以通过特定语境的语言模型LM来偏移识别的过程。但是对于端到端的模型，AM、PM、以及LM被整合成了一个神经网络模型。此时，语境偏移对于端到端的模型十分具有挑战性，其中的原因主要有以下几个方面：

- 端到端模型只在解码时用到了文本信息，作为对比，传统的ASR系统中的LM可以使用大量的文本进行训练。因此，我们发现端到端的模型在识别稀有、语境依赖的单词和短语，比如名词短语时，相较于传统模型，更容易出错。
- 端到端的模型考虑到解码效率，通常在beam search解码时的每一步只保有少量的候选词（一般为4到10个词），因此，稀有的单词短语，比如依赖语境的n-gram（n元词组），很有可能不在beam中。

**Shallow fusion (浅融合)：**将独立训练的语境n-gram 语言模型融入到端到端模型中，来解决语境建模的问题。将端到端的输出得分与一个外部训练的语言LM得分在beam search时进行融合：

<img src="readme.assets/image-20211203091705630.png" alt="image-20211203091705630" style="zoom:50%;" />

- 但是他们的方法对于专有名词处理得比较差，专有名词通常在beam search时就已经被剪裁掉了，因此即使加入语言模型来做偏移，也为时已晚。
- 因为这种偏移通常在每个word生成后才进行偏移，而beam search在grapheme/wordpiece （对于英文来说，grapheme指的是26个英文字母+1空格+12常用标点。对于中文来说，grapheme指的是3755一级汉字+3008二级汉字+16标点符号） 等sub-word单元上进行预测。

#### Shallow-Fusion End-to-End Contextual Biasing

- 首先，为了避免还没使用语言模型进行偏移，专有名词就被剪枝掉了，我们探索在sub-word单元上进行偏移。
- 其次，我们探索在beam 剪枝前使用contextual FST。
- 第三，因为语境n-gram通常和一组共同前缀(“call”, “paly”)一起使用，我们也去探索在shallow fusion时融合这些前缀
- 最后，为了帮助专有名词的建模，我们探索了多种技术去利用大规模的文本数据。无监督（NER）+有监督，合成的语音+加噪的语音

偏移的级别：

- 对grapheme进行偏移
- 对wordpiece进行偏移

> Subword算法：[深入理解NLP Subword算法：BPE、WordPiece、ULM - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/86965595)
>
> - 可用sentencepiece库

> Google, 2019

### Contextual LAS (CLAS)

#### Deep context

end-to-end contextual speech recognition

在ASR中，一个用户说话的内容取决于他所处的上下文，通常这种上下文可以由一系列的n-gram单词所代表。在ASR中，一个用户说话的内容取决于他所处的上下文，通常这种上下文可以由一系列的n-gram单词所代表。

我们的方法是首先将每个短语映射成固定维度的词嵌入，然后采用一个attention注意力机制在模型输出预测的每一步去摘要可用的上下文信息。我们的方法可以被看成是流式关键词发现技术[3]的一个泛化，即允许在推理时使用可变数量的上下文短语。我们提出的模型在训练的时候不需要特定的上下文信息，并且也不需要对重打分的权重进行仔细的调整，仍然能融入OOV词汇。

<img src="readme.assets/image-20211203094629203.png" alt="image-20211203094629203" style="zoom:50%;" />

#### EESEN

END-TO-END SPEECH RECOGNITION USING DEEP RNN MODELS AND WFST-BASED DECODING

在这个工作中，声学模型的建模是利用RNN去预测上下文无关的音素或者字符，然后使用CTC去对齐语音和label。

这篇文章与众不同的一个点是基于WFST提出了一种通用的解码方法，可以在CTC解码的时候融入词典和语言模型。CTC labels、词典、以及语言模型被编码到一个WFST中，然后合成一个综合的搜索图。这种基于WFST的方式可以很方便地处理CTC里的blank标签和进行beam search。

解码速度大大加快了。这种加速来源于状态数量的大幅减少。

#### Contextual speech recognition

Contextual speech recognition with difficult negative training examples

The main idea is to focus on proper nouns (e.g., unique entities such as names of people and places) in the reference transcript, and use phonetically similar phrases as negative examples, encouraging the neural model to learn more discriminative representations.

> [stevenhillis/awesome-asr-contextualization: A curated list of awesome papers on contextualizing E2E ASR outputs (github.com)](https://github.com/stevenhillis/awesome-asr-contextualization)
>
> [语境偏移如何解决？专有领域端到端ASR之路（一）-云社区-华为云 (huaweicloud.com)](https://bbs.huaweicloud.com/blogs/269842)
>
> [tensorflow/lingvo: Lingvo (github.com)](https://github.com/tensorflow/lingvo)

#### Phoebe

pronunciation-aware contextualization for end-to-end speech recognition

传统的ASR系统使用字典的方式使得这个系统对于未知的稀有词汇有更好的适应性；但是端到端的ASR系统对于稀有词汇的适用性较差。

Unlike CLAS, which accepts only the textual form of the bias phrases, the proposed model also has access to the corresponding phonetic pronunciations, which improves performance on challenging sets which include words unseen in training.

![image-20211203102508014](readme.assets/image-20211203102508014.png)

#### Phoneme-Based 

Phoneme-Based Contextualization for Cross-Lingual Speech Recognition in End-to-End Models

 The problem is exacerbated when biasing towards proper nouns in foreign languages, e.g., geographic location names, which are virtually unseen in training and are thus out-of-vocabulary (OOV). 

While grapheme or wordpiece E2E models might have a difficult time spelling OOV words, phonemes are more acoustically salient and past work has shown that E2E phoneme models can better predict such words.

#### Joint Grapheme and Phoneme Embeddings for Contextual End-to-End ASR,facebook

In this work, we improve the CLAS approach by proposing several new strategies to extract embeddings for the contextual entities. We compare these embedding extractors based on graphemic and phonetic input and/or output sequences and show that an encoder-decoder model trained jointly towards graphemes and phonemes out-performs other approaches

<img src="readme.assets/image-20211203104528890.png" alt="image-20211203104528890" style="zoom:50%;" />

<img src="readme.assets/image-20211203104611297.png" alt="image-20211203104611297" style="zoom:50%;" />

#### Tree-constrained Pointer Generator for End-to-end Contextual Speech Recognition

<img src="readme.assets/image-20211203133849322.png" alt="image-20211203133849322" style="zoom:50%;" />

### Contextual Transducer ("RNNTs")

#### CONTEXTUAL RNN-T FOR OPEN DOMAIN ASR

While this has some nice advantages, it limits the system to be trained using only paired audio and text. Because of this, E2E models tend to have difficulties with correctly recognizing rare words that are not frequently seen during training, such as entity names. 

we propose modifications to the RNN-T model that allow the model to utilize additional metadata text with the objective of improving performance on these named entity words.

![image-20211203105924801](readme.assets/image-20211203105924801.png)

#### MultiState encoding with end-to-end speech rnn transducer network

In this paper, we propose a technique for incorporating contextual signals, such as intelligent assistant device state or dialog state, directly into RNN-T models.

<img src="readme.assets/image-20211203125806769.png" alt="image-20211203125806769" style="zoom:50%;" />

#### DEEP SHALLOW FUSION FOR RNN-T PERSONALIZATION

 In this work, we present novel techniques to improve RNN-T’s ability to model rare WordPieces, infuse extra information into the encoder, enable the use of alternative graphemic pronunciations, and perform deep fusion with personalized language models for more robust biasing. 

#### Contextualized Streaming End-to-End Speech Recognition with Trie-Based Deep Biasing and Shallow Fusion

We address these limitations by proposing a novel solution that combines shallow fusion, trie-based deep biasing, and neural network language model contextualization.

<img src="readme.assets/image-20211203132727503.png" alt="image-20211203132727503" style="zoom:50%;" />

#### CONTEXT-AWARE TRANSFORMER TRANSDUCER FOR SPEECH RECOGNITION

Specifically, we propose a multi-head attention-based context-biasing network, which is jointly trained with the rest of the ASR sub-networks. 

We explore different techniques to encode contextual data and to create the final attention context vectors. 

We also leverage both BLSTM and pretrained BERT based models to encode contextual data and guide the network training.

![image-20211203133551330](readme.assets/image-20211203133551330.png)

### on-device

#### FAST CONTEXTUAL ADAPTATION WITH NEURAL ASSOCIATIVE MEMORY FOR ON-DEVICE PERSONALIZED SPEECH RECOGNITION 

fast contextual adaptation has shown to be effective in improving Automatic Speech Recognition (ASR) of rare words and when combined with an on-device personalized training, it can yield an even better recognition result. However, the traditional re-scoring approaches based on an external language model is prone to diverge during the personalized training. In this work, we introduce a model-based end-to-end contextual adaptation approach that is decoder-agnostic and amenable to on-device personalization

![image-20211203134042928](readme.assets/image-20211203134042928.png)

## Training Label Error

#### Investigation of Training Label Error Impact on RNN-T

数据标注错误对RNN-T影响Training Label Error Impact on RNNT

训练数据的标注错误对模型性能表现影响程度的研究在图像领域较多，在识别领域的研究较少。识别领域的训练数据标注错误主要分为三类：deletion ， insertion 和substitution。在GMM-HMM模型时代，数据的insertion错误对声学模型影响较大。本文主要研究以上三种错误对端到端的语音识别模型RNN-T的影响程度以及各种减缓错误的影响策略。

缓解标注错误的通常策略

   a) data based: 数据清洗data filtering or selection  

​    b) model capacity based: 增大模型参数量 increase model or data size

   c) optimization processs based: regularization(dropout, specaugment)

## System measuring fairness

#### Towards Measuring Fairness in Speech Recognition: Casual Conversations Dataset Transcriptions

开源一个带诸多metadata属性(性别，年龄，肤色等等）的闲聊Casual Conversations语音测试集，并使用该测试集对ASR系统进行fairness评估，发掘更多影响ASR效果的诸多因素，为开发更加鲁棒的ASR系统做贡献

## on-device

#### How to make on-device speech recognition practical

- [How to make on-device speech recognition practical - Amazon Science](https://www.amazon.science/blog/how-to-make-on-device-speech-recognition-practical)



## CTC

#### Why does CTC result in peaky behavior

- we prove that a feed-forward neural network trained with CTC from uniform initialization converges towards peaky behavior with a 100% error rate.



## Augment data

#### VOICE CONVERSION CAN IMPROVE ASR IN VERY LOW-RESOURCE SETTINGS

- voice conversion (VC) has been proposed to improve speech recognition systems in low-resource languages by using it to augment limited training data.



## Pretraing model

#### UNSUPERVISED CROSS-LINGUAL REPRESENTATION LEARNING FOR SPEECH RECOGNITION

- multi-language
- wav2vec 2.0

#### A COMPARATIVE STUDY ON TRANSFORMER VS RNN IN SPEECH APPLICATIONS

- experiment

## Metric

#### Comparison of Subword Segmentation Methods for Open-vocabulary ASR using a Difficulty Metric

- bpe, char, and word

#### Syllable-Based Sequence-to-Sequence Speech Recognition with the Transformer in Mandarin Chinese

- CI-phonemes vs Syllables

> [gentaiscool/end2end-asr-pytorch: End-to-End Automatic Speech Recognition on PyTorch (github.com)](https://github.com/gentaiscool/end2end-asr-pytorch)

## cantonese asr

#### code

- kaldi: [manestay/cantonese-asr-kaldi: kaldi repo for cantonese-sr (github.com)](https://github.com/manestay/cantonese-asr-kaldi)

> dataset:  AISHELL-2 for Mandarin and BABEL for Cantonese.

- tensorflow: [kathykyt/cantonese_ASR: https://windfat.com (github.com)](https://github.com/kathykyt/cantonese_ASR)

- pre-training: [jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn · Hugging Face](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn)[chutaklee/CantoASR: Fine-tuning Wav2Vec2.0 on Common Voice(zh-HK) (github.com)](https://github.com/chutaklee/CantoASR)
- [fairseq/examples/wav2vec/unsupervised at main · pytorch/fairseq (github.com)](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec/unsupervised)

> dataset: commonvoice_HK
>
> [Wav2vec 2.0: Learning the structure of speech from raw audio (facebook.com)](https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/)

- some package
  - [scottykwok/cantonese-selfish-project: Cantonese Selfish Project 廣東話自肥企劃 at PYCON HK 2021 (github.com)](https://github.com/scottykwok/cantonese-selfish-project/)

## 多模态语音识别

#### Attention-based Audio-Visual Fusion for Robust Automatic Speech Recognition

<img src="readme.assets/image-20211201201943154.png" alt="image-20211201201943154" style="zoom:50%;" />

- an audio-visual fusion strategy 

> dataset: TCD-TIMIT
>
> code: [georgesterpu/avsr-tf1: Audio-Visual Speech Recognition using Sequence to Sequence Models (github.com)](https://github.com/georgesterpu/avsr-tf1)





工具：

- [hirofumi0810/neural_sp: End-to-end ASR/LM implementation with PyTorch (github.com)](https://github.com/hirofumi0810/neural_sp)
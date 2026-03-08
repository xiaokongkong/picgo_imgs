@[TOC]



课程内容：如何将预训练系统 如GPT3，转换为像 ChatGPT 这样的系统？



![2025 Lecture 15 - RLHF Alignment_page_2](https://cdn.jsdelivr.net/gh/xiaokongkong/picgo_imgs/2025%20Lecture%2015%20-%20RLHF%20Alignment_page_2.png)



GPT4 模型可以跟踪非常长的代码块（比如嵌套复合指令），然后结合其编码能力生成输出。现在 chatGPT可以同时处理10条指令。

![](https://cdn.jsdelivr.net/gh/xiaokongkong/picgo_imgs/2025%20Lecture%2015%20-%20RLHF%20Alignment_page_3.png)

> * 从安全角度：模型可能会被滥用，例如被用于诈骗
> * 从内容审核角度：人们是否愿意付费使用。作为实用产品（愿意！）；作为充满恐怖毒害的系统（不愿意！）

![2025 Lecture 15 - RLHF Alignment_page_4](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_4.png)

今天的目标：尝试为语言模型建立更严格的控制。

在预训练阶段，将各种能力整合到模型中；预训练完成后，模型在参数范围内完成推理和回答问题等任务，但并非开箱即用。



所以今天，我们要让模型开箱即用。我们将收集各种行为数据，训练模型执行任务。问题是

> * 数据是什么样的，如何收集
>
> * 如何利用这些数据（某些类型的数据易于使用，比如如果有专家演示，只需要训练以模仿这些演示；但如果模型输出a优于b，要如何利用这种反馈）
>
> * 如何扩展这个过程

![](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_5.png)

本讲座的结构大致遵循 instruct GPT 论文的思路。下图是，指令跟随模型三步走的示意图。

![2025 Lecture 15 - RLHF Alignment_page_6](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_6.png)

## 监督微调

> 训练数据：要模仿专家演示，必须拥有专家演示数据
>
> 已有数据后，如何适配这些数据？答案是梯度下降，但也有不足

![2025 Lecture 15 - RLHF Alignment_page_7](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_7.png)

在前一课中，已经提到多种指令数据类型，今天将讲解其中几种。

![2025 Lecture 15 - RLHF Alignment_page_8](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_8.png)

> FLAN：谷歌团队的研究成果，本质上是通过聚合多个训练数据集构建的，来自自然语言处理任务。
>
> > 有各种不同的任务，如 natural instructions v2（包含大量问答任务），T0 SF（包含对抗性问答、主题分类），所以是通过获取现有的、执行各种独立任务的自然语言处理数据集来构建的，然后合并成一个大型元数据集。
>
> Oasst：一群在线爱好者聚集起来，共同编写语言模型的指令调优数据。
>
> Alpaca：来自斯坦福

![2025 Lecture 15 - RLHF Alignment_page_9](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_9.png)

### FLAN-示例

有一些看起来像正常指令调优数据，如第1条——“为邮件写主题行”（来自 Enron email dataset），第2条——“多项选择”，第3条——“为这篇文章写亮点”，第4条——“写出描述这家餐厅的句子”（来自E2E dataset）

> 缺点是，数据不够自然，不是普通的聊天互动

![2025 Lecture 15 - RLHF Alignment_page_10](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_10.png)

### Alpaca-示例

尝试用语言模型生成指令调优数据。有一组人类撰写的指令，接着用语言模型生成更多指令(得到左边的列)，再用类似 instruct GPT 的模型填充回应（右列）。

> 缺点是，输入不够多样，输入的指令非常简短

![2025 Lecture 15 - RLHF Alignment_page_11](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_11.png)

### OpenAssistant-示例

左边是更复杂的查询，右边的回答非常详细，甚至包含引用。

> 质量很高，但是构建很困难

![2025 Lecture 15 - RLHF Alignment_page_12](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_12.png)

> 这里老师让大家做了一个互动，输入各自的答案，以生成不同的数据。结论是，大家生成的数据都很简短，

![2025 Lecture 15 - RLHF Alignment_page_13](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_13.png)

GPT-4o的回复相当不错，内容很长很详细。如果是人类回复，会需要大量精力和成本。

![2025 Lecture 15 - RLHF Alignment_page_14](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_14.png)

chatGPT 生成的数据有以下特点：

> 在长度方面差异很大
>
> 采用要点形式
>
> 风格多样

![2025 Lecture 15 - RLHF Alignment_page_15](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_15.png)

回到2023年，非常流行生成这类指令调优数据集，输入和输出的长度差异很大。



输入长度反映了任务的复杂度，输出长度衡量了回答的深度。

![2025 Lecture 15 - RLHF Alignment_page_16](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_16.png)

如果使用人类标注，人们普遍偏好列表形式；如果AI标注，也偏好列表形式。

这会令人担忧，因为需要优化的不仅仅是回复的风格内容，还应该通过后训练减少幻觉现象，并真正提升模型能力。

![2025 Lecture 15 - RLHF Alignment_page_17](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_17.png)

尽管许多模型存在显著长度差异，大多数指令调优数据集，相比于基础模型，能带来提升。

> 聊天风格的评估，Chatbot Arena、AlpacaEval 等自动化评估都有其用武之地，有助于理解用户参与度，但基准测试同样重要。
>
> 因为在后训练时，不一定希望受到比如长度偏差等因素的过多影响，所以采用多种评估策略来避免。

![2025 Lecture 15 - RLHF Alignment_page_18](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_18.png)

### 指令调优的反直觉现象

这里有一个输入-输出对的例子。左边是要求一个关于买方垄断经济学的介绍，右边的回复里有参考文献。

假设有一个模型，微调时，以左边作为输入，右边作为输出。

这个过程会有两件事同时进行：（1）建立关联：将垄断与该引用联系起来；（2）泛化的行为：如果你问我一个复杂概念，我应该在输出中添加引用。



第一件事传授新知识，但第二件事在教模型生成虚假信息。

> 如果你这样做，会让模型产生幻觉。因为模型并不具备回答问题的知识，你却强迫它回答问题。它将学会的是，以某种抽象形式掌握知识，以及需要编造一些内容来呈现答案。

![2025 Lecture 15 - RLHF Alignment_page_19](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_19.png)

学习需要添加引用，这件事情本身没问题，要解决的是记忆问题或工具使用问题。

> on-policy RL 为什么重要？你需要知道模型已知的内容，并只教授它能理解的东西，以避免幻觉。而且当它遇到一些不知道的事实时，要修改微调数据，让模型学会说“我不知道这个事实”，而不是强迫模型回答。

![2025 Lecture 15 - RLHF Alignment_page_20](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_20.png)

你可以拥有一个完全正确且内容丰富的指令调优数据集，但实际上可能不利于你的语言模型，因为它会让模型学会编造事实，以匹配原有的知识深度。

> 这个论点解释了为什么你需要非常谨慎地处理蒸馏数据。当教师模型比学生模型更强时，还需要非常专业的标注数据，因为人类可能比模型更具备专业知识，你需要确保模型在不知情时能正确拒绝。



![2025 Lecture 15 - RLHF Alignment_page_21](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_21.png)

幻觉问题，不能只通过指令调优解决，还需要考虑安全问题，并分析相关权衡。



语言模型需要设置安全边界，因为直接面向用户部署，能力很强，可能被用于传播错误信息，或者生成诈骗、垃圾信息等内容。因此需要对模型进行安全调优。



![2025 Lecture 15 - RLHF Alignment_page_22](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_22.png)

即使少量安全调优数据，融入指令训练过程，也能使模型显著更安全。这和指令调优类似，如果模型训练充分，少量指令调优数据就能带来显著效果。但这并不意味着足够了，只是能达到一个不错的程度。



![2025 Lecture 15 - RLHF Alignment_page_23](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_23.png)

安全调优的核心权衡是，拒绝与不过度拒绝之间的权衡。



如果你有不安全的回应，你希望安全调优模型直接拒绝回答；

但有一些实际安全但看起来不安全的回应，比如 how to **kill** a python process。我们都知道这是合理的问题，但如果模型对英语理解不够深入，认为 **kill** 是个危险的词，可能就会拒绝。

所以，如何让模型理解这种细微差别，是很难的。



![2025 Lecture 15 - RLHF Alignment_page_24](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_24.png)

即使有500个例子，也能让模型遵化部分安全准则

![2025 Lecture 15 - RLHF Alignment_page_25](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_25.png)

综合来看，

1. 指令微调非常强大。

> 如果用一个相对标准的指令微调数据集（比如 open hermes 或 open assistant），并取一个基础模型，用合理的参数进行微调，就会得到类似 Llama 或 ChatGPT 的模型。效果不会完全一样，仍然需要大量额外的工作来优化，但能取得显著进展。

2. 高质量数据的本质及其复杂。

3. 即使少量数据，也能产生巨大作用，改变模型的行为方式。

![2025 Lecture 15 - RLHF Alignment_page_26](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_26.png)

如何微调？

只需输入指令和 response，然后进行梯度下降。但是如果你拥有大量的的计算资源和数据，就可以大幅扩展这个过程。

![2025 Lecture 15 - RLHF Alignment_page_27](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_27.png)

现在指令微调和预训练的界限越来越模糊。指令微调数据仍然是一个序列，可以直接加入预训练过程。

> 1. 进行纯预训练
>
> 2. 将指令微调数据混入预训练
>
>    >  在预训练的后期，尤其是是逐渐降低学习率时，开始加入大量高质量数据或指令微调数据。
>    >

>
>3. 最后可能再进行一轮短暂的第二轮指令微调，但这轮可能规模更小，因为你的大部分数据已经进入了第二阶段，这个过程也被称为中期训练(mid-training)。这很棒，因为它允许你在不出现灾难性遗忘问题的情况下扩展规模。
>
>   
>
>   > 优点：可以从数据中获得更大收益，因为它更深入地整合到预训练中。
>   >


![2025 Lecture 15 - RLHF Alignment_page_28](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_28.png)

### 两阶段训练过程

> 第一阶段：纯预训练，如左边的饼图，都是预训练数据集（common crawl、code pretrain、dolma等）
>
> 第二阶段：被称为衰减阶段 [【cs336学习笔记】[第11课]如何用好scaling law（中提到了WSD，对应了预热-衰减-稳定阶段）](https://blog.csdn.net/m0_37586991/article/details/151230489?spm=1001.2014.3001.5501)，如右边的饼图，采用的是 wikipedia 等高质量数据，混合了预训练的内容

两阶段训练：利用损失大幅下降，将模型引导至正确的模式。

> Q：针对灾难性遗忘吗？
>
> A：灾难性遗忘是两阶段训练的动机。当你有如此多的 SFT 数据，权衡会很难，除非采用类似预训练和后训练混合的方式，此时必须考虑正则化、微小步长，防止破坏预训练效果。
>
> Q：对解决引用问题有帮助吗？
>
> A：没有帮助。唯一能解决引用问题的是，你知道模型知道哪些事实。确保模型在展示 SFT 数据前始终掌握这些引用事实。
>
> 

![2025 Lecture 15 - RLHF Alignment_page_29](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_29.png)



> Q：预训练可以传授模型新知识，指令调优是否也可以？
>
> A：如果指令调优规模足够大，且有足够的多样性，会传授知识；但如果规模较小，也不是中期训练的形式，就很难传授各种知识。

## RLHF-强化学习人类反馈



![2025 Lecture 15 - RLHF Alignment_page_30](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_30.png)

SFT：

存在一个参考分布 $p^*(y|x)$，类似于互联网数据和标注数据的混合，我们的目标是模仿 $p^*$。这是纯粹的生成模型



RLHF：

不关心匹配任何分布，概率视角并没有完全消失，但是要谨慎采用。

我们寻找的是某种策略 $\hat{p}(y|x)$，能够最大化奖励 $R(y, x)$。所以，大模型不一定是某个潜在分布的模型，而是能给我们带来丰厚奖励的策略。

> 为什么要做RLHF？
>
> 1. 和 SFT 有关，为了进行模仿过程，需要从  $p^*$ 中采样，成本会很高
> 2. RLHF 只需要获取奖励的测量结果，而 SFT 的数据是很贵的。

![2025 Lecture 15 - RLHF Alignment_page_31](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_31.png)

不同阶段可能遇到的成本，SFT 可能会非常昂贵。



![2025 Lecture 15 - RLHF Alignment_page_32](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_32.png)

有一个有趣的现象，人们对于什么是好的，并不总是与自己意见一致。如果你让某人撰写摘要，再让他比较自己的摘要和大模型生成的摘要，有很大比例的人会更喜欢语言模型生成的结果。

因此，验证不仅比生成成本更低，而且验证的质量可能比生成更高。也就是存在生成者和验证者之间的差异。



![2025 Lecture 15 - RLHF Alignment_page_33](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_33.png)



### 总览

> 三个方面
>
> * 数据
>   * 如何收集数据
>   * 应该担心哪些事情
> * 如何做RLHF？
>   * PPO
>   * DPO
> * RLHF 的副作用



![2025 Lecture 15 - RLHF Alignment_page_34](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_34.png)

当进行 InstructGPT 流程的第二部分时，首先让模型生成自己的输出（在强化学习中被称为 rollouts ），然后进行比较，这里展示了四个输出ABCD，但标准设置是2个。给定A和B，判断A是否比B更好。根据成对反馈，训练一个奖励模型，为每个输出赋予一个标量值，再进行强化学习，希望模型能最大化这些奖励。



![2025 Lecture 15 - RLHF Alignment_page_35](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_35.png)

### 数据

如何收集成对的奖励反馈数据？

开发一个网页应用，有两个不同的 AI 回答，有一个复选框，让用户选择更好的回答。



![2025 Lecture 15 - RLHF Alignment_page_36](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_36.png)

#### 标注指南示例

对于标注人员，工作就是评估这些输出，确保它们有帮助、真实且无害，

![2025 Lecture 15 - RLHF Alignment_page_37](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_37.png)

谷歌 bard 的真实标注指南：

> * 左上角：有用性，应该回应用户的意图、遵守任何要求，不要包含误导信息
>
> * 有一个风格框，说明哪些风格是优劣的。
>
> * 针对不同回应有不同的评分量表



![2025 Lecture 15 - RLHF Alignment_page_38](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_38.png)

InstructGPT：通过 scale 和 upwork，从大约 40 人那里收集数据，



![2025 Lecture 15 - RLHF Alignment_page_39](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_39.png)

> 老师在这里进行了第二次互动，给定一个问题（写一份 sharad kumar 的自传），并给出两个option（内容较短和较长）进行选择。实际上，较长的版本带有幻觉；较短的版本是简化了较长的版本，并去除幻觉。从投票结果来看，较长版本有更多选票。
>
> >
>
> 因为每个标注员只有很短的时间，所以很难获得高质量、可验证的标注员

![2025 Lecture 15 - RLHF Alignment_page_40](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_40.png)

RLHF 和数据 - 众包

> 众包的复杂性
>
> * 很难找到真正高质量、可验证的b标注者
> * 很难让他们真正检查正确性
> * 必须谨慎使用 GPT4……（标注者会用 GPT4 来代替自己做出判断）

所以，尽管 成对反馈 比 监督数据 更容易收集，仍存在明显的问题。



![2025 Lecture 15 - RLHF Alignment_page_41](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_41.png)

关于众包的问题已有很多论述。

如果你试图将这份工作外包给其他国家，会产生以下问题，还有定价方面的困扰。



![2025 Lecture 15 - RLHF Alignment_page_42](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_42.png)

偏见和安全：对模型对齐很重要！

从某种意义上来说，RLHF和对齐是在整个流程的最后阶段进行的。由于位于流程末端，会对模型行为有很强的影响。其中有一篇论文，探讨大模型语言的主观意见如何与不同人群的观点保持一致。

> 有一个有趣的现象是，InstructGPT 变得更符合东南亚宗教的观点，查看附录中标注人员的国籍，发现是菲律宾（22%）、孟加拉国（22%）和美国人（17%）。



![2025 Lecture 15 - RLHF Alignment_page_43](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_43.png)

也有人指出，不同的标注员，关注的点不同。

两种类型的标注员：

> 1. 非常有动力，用引号正确判断事物
> 2. 不太关注事实，更注重格式正确。

所以取决于不同的标注员，即使问一样的问题，你也会得到不同的反馈。



![2025 Lecture 15 - RLHF Alignment_page_44](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_44.png)

于是，越来越多的人转向 AI 反馈，如果你尝试从 GPT-4 获得成对反馈，会和 GPT-4 响应的速率有非常高的一致性。

> 左图，纵轴是人类评估的结果。
>
> 右图，蓝色方框表示，对AI反馈的结果，GPT-4和人类之间的一致性大致相同

![2025 Lecture 15 - RLHF Alignment_page_45](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_45.png)

> * Ultrafeedback：用于 off-policy RLHF。UltraFeedback 是由 OpenBMB 团队发布的大规模、细粒度、多样化偏好数据集，包含约 6.4 万个多源提示与 25.6 万个多模型响应，经 GPT-4 从指令遵循、真实性、诚实性、帮助性四维度进行数值与文本双反馈标注，可构建约 34 万个比较对，核心用于训练奖励模型（如 UltraRM）与批评模型，支撑强化学习从人类反馈（RLHF）领域的对齐研究。
> * Zephyr 7b：是 Hugging Face H4 基于 Mistral-7B-v0.1 模型，通过 Direct Preference Optimization（DPO）技术微调而成的 70 亿参数语言模型，在 MT Bench 和 AlpacaEval 等基准测试中表现出色。
> * Tulu3：是基于 Meta 的 Llama 3.1 模型开发的，它代表了开放后训练的新标准，通过引入可验证奖励的强化学习方法等，在多个任务上超越了 Llama 3.1 模型。



![2025 Lecture 15 - RLHF Alignment_page_46](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_46.png)

以上方法都源自于 [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/pdf/2212.08073)，关于 AI 反馈被用于这种对齐过程。



![2025 Lecture 15 - RLHF Alignment_page_47](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_47.png)

#### 长度效应

很多人看到更长的回复，会觉得更详细。这其实是一种偏见。所以，人们认为更好的模型，可能只是因为回复更长。而AI反馈似乎会让模型普遍变得更长。

> off-policy：单独收集这类成对反馈数据，也就是说，他们不是从你的模型输出中收集的。也许你的模型会参与其中，但这些数据并非来自于你的模型。off-policy像是在告诉你那些你尚未涉足领域的整体情况。
>
> on-policy：告诉你如何提升自己。
>
> 详细区别可见：[通俗易懂地解释 On-Policy 和 Off-Policy 的区别](https://blog.csdn.net/m0_37586991/article/details/151971002?sharetype=blogdetail&sharerId=151971002&sharerefer=PC&sharesource=m0_37586991&spm=1011.2480.3001.8118)

![2025 Lecture 15 - RLHF Alignment_page_48](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_48.png)

> Q：如果模型获取的信息量来自于自身的输出，也就是让模型自我反馈，是否有一个“做多少次”的启发式方法？以及如果通过抽样来训练模型，结果会改变吗？
>
> A：从信息理论来看，界限非常高。因为模型吸收了整个预训练语料库，这些信息可能存储在模型的某个地方。而且取决于你如何提示，可能会得到一个表现出色的模型。因此，根据你如何将模型用作“自我改进循环”的一部分，可以从模型中提取更多的能力，上限是未知的，因为输入数据量非常庞大。目前确实有论文研究“自我改进”能带来多大帮助，但最终得靠实践证明。



### 如何做RLHF？

我们现在有了（高质量的）成对反馈数据收集流程吗？
我们如何调整模型以利用成对反馈？
第一部分：PPO——原始且非常复杂的方法（简要版）
第二部分：DPO——全新且易于理解的方法

![RLHF](https://github.com/xiaokongkong/picgo_imgs/blob/main/2025%20Lecture%2015%20-%20RLHF%20Alignment_page_49.png)

![2025 Lecture 15 - RLHF Alignment_page_49](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_49.png)

目标是，找到一个策略（policy），来最大化奖励（reward）

所以接下来讲解两个问题：什么是奖励？最大化的过程是什么？



![2025 Lecture 15 - RLHF Alignment_page_50](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_50.png)

* $r_\theta(x,y)$ 是奖励函数

* $log(\pi^{RL}_\phi(y|x)/\pi^{SFT}(y|x))$ 是 RL策略 除以 SFT模型输出 的对数比，

* $\pi^{RL}_\phi(y|x)$ 是 RL策略 与原始 SFT 模型之间的 KL散度，含义是，在进行 RL 时，不要离最初的 SFT 模型太远。

* $\gamma$ 因子所在的式子，代表在进行强化学习时，也要持续进行预训练。这样做，就不会发生灾难性遗忘。

![2025 Lecture 15 - RLHF Alignment_page_51](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_51.png)

奖励是什么？

存在一种被假设出来的世界模型，每个序列都关联着一个标量值 R，我们无法直接观测到 R 是什么。

> 当有一个人进行A和B的成对比较打分时，其实在比较那两个序列各自的奖励值，基于差异进行抛硬币。这就是基于两个奖励差异的逻辑回归模型，每个序列都有一个奖励值。
>
> 这就是用来建模人类偏好的 Bradley terry luce 模型。



所以优化奖励值时，我们在做的就是输出具有最高 R 值的序列，但 R 无法直接观测到，只能通过模型参数 $\theta$ 观测到带有噪声的成对比较结果。



![2025 Lecture 15 - RLHF Alignment_page_52](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_52.png)

### PPO

我们想要优化某个策略的奖励值，那么优化一个东西的好方法是什么呢？计算梯度，即梯度下降。

Attempt 1: Policy gradients (variances are too high：算法训练过程中，参数更新的方差过大，会导致训练不稳定)

$$ \nabla_{\theta} E_{p_{\theta}}[R(z)]=E_{p_{\theta}}\left[\mathrm{R}(\mathrm{z})  * \nabla_{\theta} \log p_{\theta}(z)\right]$$

> 奖励的期望值 乘以 $p_\theta$ 的梯度。相当于说，对 $p_\theta$ ，想要最大化它的概率。
>
> 如果奖励是正的，就提高那些概率的权重；如果是负的，就降低那些概率的权重。这是策略梯度定理。
>
> 从 $R(z)$ 中减去任何某种状态相关的变量，梯度也仍然是正确的。这意味着，我可以在减去任何我想要的基线值后，将这个奖励重新写成另一种形式。如下面 TRPO 中 maximize的部分。


Attempt 2: TRPO (Linearize the problem around the current policy, 围绕当前策略对问题进行线性化)

$$\begin{array}{cl}
\underset{\theta}{\operatorname{maximize}} & \hat{\mathbb{E}}_{t}\left[\frac{\pi_{\theta}\left(a_{t} \mid s_{t}\right)}{\pi_{\theta_{\text {old }}}\left(a_{t} \mid s_{t}\right)} \hat{A}_{t}\right] \\
\text { subject to } & \hat{\mathbb{E}}_{t}\left[\operatorname{KL}\left[\pi_{\theta_{\text {old }}}\left(\cdot \mid s_{t}\right), \pi_{\theta}\left(\cdot \mid s_{t}\right)\right]\right] \leq \delta
\end{array}$$



> 不直接使用奖励，而是看一个叫“优势”的概念，优势基本是奖励的方差降低版本。
>
> 在 $p_\theta$ 采样一次后，进行多次梯度更新。从一次 rollout 中采样，并且几乎变成了 off-policy 。
>
> 为了实现这一点，必须进行重要性加权校正。因为我进行的步骤越多，原始样本就变得越陈旧。
>
> 这就是TRPO。对进行的所有梯度更新进行校正，并约束自己保持接近旧策略


Attempt 3: PPO (Clip the ratios at some eps，将比值裁剪在某个 ε 范围内)

$$\operatorname{clip}\left(\frac{\pi_{\theta}(a \mid s)}{\pi_{\theta_{k}}(a \mid s)}, 1-\epsilon, 1+\epsilon\right) A^{\pi_{\theta_{k}}}(s, a))$$

> PPO说，与其明确地约束自己保持接近旧策略，不如使用 KL 约束，直接 裁剪 概率比率，这会自然地促使模型保持接近原始策略。



![2025 Lecture 15 - RLHF Alignment_page_53](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_53.png)

在开发研究领域，人们关注的是，能否去掉 PPO？因为 PPO 非常复杂。



一些合理的方法是

> * 对 pairs 进行SFT，但对于每个 pair，我们可以假定一个“好”的 token 对应选定的优质输出，以及一个“坏”的 token 对应坏的输出。在生成时，只以“好”为条件。但效果不是很好。
>
> * 只用偏好的输出来训练模型。效果也不是很好
> * 用一个奖励模型，从结果中采样出最好的一个，再用最好的结果进行训练。效果还行，但可能也没那么好



![2025 Lecture 15 - RLHF Alignment_page_54](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_54.png)

### DPO

DPO 消除了 PPO 的很多复杂性，而且效果相对不错：

> * 移除奖励模型（用于计算优势）
>
> * 移除任何 on-policy 的东西（例如刚才的重要性加权的概念）

DPO的做法：

> * 对好的结果的对数损失进行梯度更新
> * 对坏的结果的对数损失进行负梯度更新



![2025 Lecture 15 - RLHF Alignment_page_55](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_55.png)

DPO 公式推导

目标是优化 $$\max _{\pi_{\theta}} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y \mid x)}\left[r_{\phi}(x, y)\right]-\beta \mathbb{D}_{\mathrm{KL}}\left[\pi_{\theta}(y \mid x) \| \pi_{\mathrm{ref}}(y \mid x)\right]$$

> $$\max _{\pi_{\theta}} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y \mid x)}\left[r_{\phi}(x, y)\right]$$ 是奖励项
>
> $$\beta \mathbb{D}_{\mathrm{KL}}\left[\pi_{\theta}(y \mid x) \| \pi_{\mathrm{ref}}(y \mid x)\right]$$ ：KL散度，使策略 $\pi_{\theta}(y \mid x)$ 接近参考策略 $\pi_{\mathrm{ref}}(y \mid x)$



假设策略 $\pi_{\theta}(y \mid x)$ 不是神经网络，是任意一个类型的函数，那么最优策略的形式是

$$\pi_{r}(y \mid x)=\frac{1}{Z(x)} \pi_{\mathrm{ref}}(y \mid x) \exp \left(\frac{1}{\beta} r(x, y)\right)$$

> $\exp (\frac{1}{\beta} r(x, y))$ 是奖励的指数形式，并且乘以参考策略 $ \pi_{\mathrm{ref}}(y \mid x)$



通过求解 $r(x, y)$ 来求出隐含的奖励。具体是通过两边取对数，将 $r(x, y)$ 放到左边，其余放到右边

$r(x, y)=\beta \log \frac{\pi_{r}(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}+\beta \log Z(x)$

DPO 的巧妙之处在于，对于任何策略，与其考虑策略本身，不如考虑奖励。因为在非参数假设下，两者是等价的



![2025 Lecture 15 - RLHF Alignment_page_56](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_56.png)





We can now optimize the implied reward as a reward model via the Stiennon objective

$\operatorname{loss}\left(r_{\theta}\right)=-E_{\left(x, y_{0}, y_{1}, i\right) \sim D}\left[\log \left(\sigma\left(r_{\theta}\left(x, y_{i}\right)-r_{\theta}\left(x, y_{1-i}\right)\right)\right)\right] \quad$：是来自 stiennon 论文的 bradley-terry 方程

$r(x, y)=\beta \log \frac{\pi_{r}(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}+\beta \log Z(x)$：DPO的等价形式



现在，将奖励R代入到目标函数中，然后最小化损失。

> 即，我要找到一个策略，使得该策略对应的隐含奖励，生成我的 pair 比较 的概率最高。
>
> 这样，把强化学习问题转换成了一个最大似然问题，一个在概念上与预训练非常相似的问题

$\mathcal{L}_{\mathrm{DPO}}\left(\pi_{\theta} ; \pi_{\mathrm{ref}}\right)=-\mathbb{E}_{\left(x, y_{w}, y_{l}\right) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi_{\theta}\left(y_{w} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{w} \mid x\right)}-\beta \log \frac{\pi_{\theta}\left(y_{l} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{l} \mid x\right)}\right)\right]$



> 总结：
>
> 做出非参数假设，通过策略参数化奖励（把奖励重写为策略之间的比率），再用监督损失进行优化

![2025 Lecture 15 - RLHF Alignment_page_57](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_57.png)

DPO的更新有以下形式：

可以看作是一种正则化。当奖励估计错误时，会赋予更高的权重。增加好样本 $y_w$ 的似然，降低坏样本 $y_l$ 的似然。

| 强化学习算法通常可以归为：给好的东西增加权重，给坏的东西减少权重。微秒之处在于如何决定什么是好的东西，以及增加多少权重。

![2025 Lecture 15 - RLHF Alignment_page_58](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_58.png)



![2025 Lecture 15 - RLHF Alignment_page_59](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_59.png)





基本上所有开源模型发布都是用了DPO的某种变体，来完成他们的 post-training 阶段。相比PPO，更容易实现。

![2025 Lecture 15 - RLHF Alignment_page_60](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_60.png)

#### DPO变体

DPO有很多变体，下面提到的两篇是最近被人使用过的。



SimPO：只做了非常简单的两个修改，第一个是根据 response 的长度来标准化更新的大小；第二个是去掉了$\pi_{ref}$。在DPO中我们比较策略；在SimPO中我们更纯粹地观察某事物。



Length normalized DPO：保留了$\pi_{ref}$，只是按长度进行标准化。



![2025 Lecture 15 - RLHF Alignment_page_61](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_61.png)

在强化学习中，很多发现都依赖于特定的设置。根据不同的运行环境、基础模型、后训练偏好，会得出完全不同的结论。



如左图，PPO 比 DPO 更好，可能是因为它更偏向 on-policy。从61.0 -> 62.2时，展示的是DPO和PPO之间的差距。

研究人员发现，如果以一种更好更优的方式进行SFT，即抵消了PPO和DPO的所有优势，两者均无收益。（如右图）。唯一做的更好的是带有长度标准化的DPO。

| 注意：你不应该把任何单一的实验结果，必然地视为金科玉律。

![2025 Lecture 15 - RLHF Alignment_page_62](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_62.png)



### 需要注意的两点

#### 过度优化（over optimization）

即过拟合换了种说法。当你越来越多地优化策略时（类似左图的x轴，代表进行了多少强化学习），最初奖励值会越来越高，但最终，你根据人类偏好拟合的奖励模型会开始偏离真正的人类偏好。优化得越多，最终偏离得越远。

![2025 Lecture 15 - RLHF Alignment_page_63](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_63.png)

| 类似于训练集和验证集的差距，x轴代表的是训练集（拟合了一个奖励模型，并测量拟合好的奖励模型），y轴是来自真实奖励的全新样本。从期望上来说，它们衡量的是同一样东西。但在有限样本下，衡量的不是同一样东西。

过度优化会以多种方式出现，因为人类偏好有不确定性和复杂性。



有学生做了一项研究，基于人类偏好进行RLHF，图a是对有噪声版本的AI反馈进行了RLHF，可以看到明显的过度优化现象；图b是干净、无噪声的AI反馈，就没有过度优化。如果你进行微调训练，会看到类似图a的曲线。



所以，当模型在代理奖励上持续优化，你所测量的指标未必能得到真正符合人类偏好的更好模型。



![2025 Lecture 15 - RLHF Alignment_page_64](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_64.png)

| x轴代表的是强化学习的成功程度，Y轴表示真实的胜率（由AI反馈或人类反馈衡量得到，是真实的人类投票）。也就是说，思路是在强化学习上取得成功，就能在实际任务上取得成功。但事实并非如此，至少会在某个点上会人类偏好产生过拟合。





#### 模式崩溃

当我们进行强化学习时，不再处于概率世界。所以当我们进行监督微调或预训练时，做的是在某个分布上进行概率建模，也就是进行某种分布匹配。



但是RLHF完全不同，它是一个策略，不一定存在底层分布，通常会得到校准程度低得多的模型。

如左上角的图，在温度设置为1时，模型表现出更过度自信的行为。

这可能也还ok，因为校准并不是你设定的奖励的一部分。但你如果将这些模型视为经过校准的概率模型，就得非常小心。



![2025 Lecture 15 - RLHF Alignment_page_65](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_65.png)



**RLHF 要点回顾：**

1. RLHF 数据收集（也）颇具难度！存在诸多干扰因素。
2. RLHF 算法比监督微调（SFT）更复杂一些 —— 尤其是近端策略优化（PPO）。
3. 需注意（过度）优化奖励所带来的影响：

![2025 Lecture 15 - RLHF Alignment_page_66](/Users/hxd/projects/xiaomi_projects/tools_code/codes/tmp_codes/csdn/pdf_images/lecture_15/2025 Lecture 15 - RLHF Alignment_page_66.png)
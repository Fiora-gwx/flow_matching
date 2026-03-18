# Paper Draft

这个目录存放 NeurIPS 稿件源码，采用双模式、单骨架组织：

- `main.tex` 默认生成中文写作用稿
- `submission_wrapper.tex` 生成匿名投稿骨架
- 两种模式共享同一套章节、标签和参考文献键

## 目录结构

- `main.tex`: 默认入口，编译中文写作用稿
- `submission_wrapper.tex`: 匿名投稿骨架入口
- `sections/`: 共享章节内容，统一通过 `\input{}` 引入
- `references.bib`: 正文实际引用到的真实文献
- `figures/`: 图表与占位图目录

## 样式文件

当前目录已经包含从官方样式包解压得到的：

- `/Users/master/Documents/project/wt-t001-ft-clock-experiment-platform/paper/neurips_2025.sty`
- `/Users/master/Documents/project/wt-t001-ft-clock-experiment-platform/paper/neurips_2025_official_template.tex`
- `/Users/master/Documents/project/wt-t001-ft-clock-experiment-platform/paper/neurips_2025_instructions.pdf`

原始压缩包内容保存在：

- `/Users/master/Documents/project/wt-t001-ft-clock-experiment-platform/paper/neurips2025_styles/Styles/`

如果未来样式文件被移走，`main.tex` 会退回到本地 `article` 构建模式，仅用于草稿预览，不代表正式投稿版式。

## 编译

当前工作区已经验证 `tectonic` 可以直接编译：

```bash
cd /Users/master/Documents/project/wt-t001-ft-clock-experiment-platform/paper
tectonic main.tex
tectonic submission_wrapper.tex
```

如果本机单独安装了 XeLaTeX，也可以使用标准流程：

```bash
cd /Users/master/Documents/project/wt-t001-ft-clock-experiment-platform/paper
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

匿名投稿骨架模式：

```bash
cd /Users/master/Documents/project/wt-t001-ft-clock-experiment-platform/paper
xelatex submission_wrapper.tex
bibtex submission_wrapper
xelatex submission_wrapper.tex
xelatex submission_wrapper.tex
```

## 说明

- 中文模式服务于写作，不用于判断 NeurIPS 页数。
- 匿名模式保留投稿版式和相同章节骨架，便于后续英文化。
- 主文已经去除内部说明、代码库说明和实验管理语言；内部协作信息只保留在 README 与源码注释中。
- 实验结果尚未补入，图表当前为论文式占位，具体替换要求写在相应 `.tex` 注释里。

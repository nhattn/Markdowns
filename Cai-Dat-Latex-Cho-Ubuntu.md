# Cài TexLive

```bash
sudo apt-get install texlive-latex-base
```

# Cài đặt gói hỗ trợ tiếng Việt

```bash
sudo apt-get install texlive-lang-other
```

# Cài đặt Gummi

```bash
sudo apt-get install gummi
```

# Tạo thử một tài liệu

```bash
tee -a /tmp/test.tex << END
> \documentclass[a4paper,11pt]{article}
> \usepackage[utf8]{vietnam}
> \usepackage{amsmath}
> \begin{document}
> \title{Dùng tiếng Việt (UTF-8) trong \LaTeX}
> \maketitle
> \section{Tiêu đề}
> \subsection{Tiểu tiêu đề}
> Một vài đoạn văn bản
> 
> Một phương trình nổi tiếng của vật lý học $ E=mc^2 $ hay phương trình bậc hai $ ax^2 + bx + c = 0 $
> \end{document}
> END
```

# Chay thử

```bash
gummi /tmp/test.tex
```

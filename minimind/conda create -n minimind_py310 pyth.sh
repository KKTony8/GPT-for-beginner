####@https://github.com/jingyaogong/minimind
##Test0
conda create -n minimind_py310 python=3.10 -y
git lfs install

git clone https://github.com/jingyaogong/minimind.git
cd minimind
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

#这一步服务器无法访问huggingface.co，所以可以本地设置端口下载
git clone https://huggingface.co/jingyaogong/MiniMind2

# 可能需要`python>=3.10` 安装 `pip install streamlit`
# cd scripts
streamlit run web_demo.py
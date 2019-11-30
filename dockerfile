FROM store/continuumio/anaconda:4.0.0
RUN git clone https://github.com/aditya140/ques_gen.git
workdir ./ques_gen
RUN pip install -r requirements.txt
CMD python train.py

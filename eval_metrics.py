from pycocoevalcap.bleu import bleu
from pycocoevalcap.meteor import meteor
from pycocoevalcap.cider import cider
from pycocoevalcap.rouge import rouge
import pandas as pd


class qgeval(object):
    def __init__(self,path="eval_results.csv"):
        self.path=path
        self.df=pd.read_csv(path)
        
    def calc_metrics(self):
        df=self.df[["Target Question","Predited Question"]]
        def to_list(x):
            return [x]
        df["Target Question"]=df["Target Question"].apply(to_list)
        df["Predited Question"]=df["Predited Question"].apply(to_list)

        x=df.to_dict()
        
        gts=x["Target Question"]
        res=x["Predited Question"]
        
        bleu_scorer=bleu.Bleu()
        meteor_scorer=meteor.Meteor()
        cider_scorer=cider.Cider()
        rouge_scorer=rouge.Rouge()
        
        bleu_score=bleu_scorer.compute_score(gts,res)
        meteor_score=meteor_scorer.compute_score(gts,res)
        cider_score=cider_scorer.compute_score(gts,res)
        rouge_score=rouge_scorer.compute_score(gts,res)
        
        s=f"Bleu1 : {bleu_score[0][0]} \nBleu2 : {bleu_score[0][1]} \nBleu3 : {bleu_score[0][2]} \nBleu4 : {bleu_score[0][3]} \n"
        s+=f"Meteor : {meteor_score[0]}\n"
        s+=f"Cider : {cider_score[0]}\n"
        s+=f"ROUGE : {rouge_score[0]}\n"
        print(s)
        with open("metrics.txt",'w') as f:
            f.write(s)
        



if __name__=="__main__":
    evaluator=qgeval()
    evaluator.calc_metrics()
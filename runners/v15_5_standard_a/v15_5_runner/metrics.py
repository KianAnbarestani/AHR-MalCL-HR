import numpy as np

def confusion_metrics(y_true,y_pred,C):
    cm=np.zeros((C,C),dtype=np.int64);np.add.at(cm,(np.asarray(y_true,int),np.asarray(y_pred,int)),1);support=cm.sum(1);tp=np.diag(cm);pred=cm.sum(0)
    recall=np.divide(tp,support,out=np.zeros(C,float),where=support>0);precision=np.divide(tp,pred,out=np.zeros(C,float),where=pred>0);f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(C,float),where=(precision+recall)>0)
    return cm,{'accuracy':float(tp.sum()/max(cm.sum(),1)),'macro_f1':float(f1.mean()),'weighted_f1':float(np.average(f1,weights=support)),'balanced_accuracy':float(recall.mean()),'per_class_precision':precision,'per_class_recall':recall,'per_class_f1':f1,'support':support}

def continual_metrics(matrix,supports):
    a=np.asarray(matrix,float);n=len(a);stage_tm=[];stage_sw=[]
    for t in range(n):
        vals=a[t,:t+1];stage_tm.append(float(np.mean(vals)));w=np.asarray(supports[:t+1],float);stage_sw.append(float(np.average(vals,weights=w)))
    final=a[-1];old=range(n-1)
    return {'final_task_macro_accuracy':float(np.mean(final)),'tm_aia':float(np.mean(stage_tm)),'sw_aia':float(np.mean(stage_sw)),
      'forgetting':float(np.mean([np.max(a[u:n-1,u])-final[u] for u in old])),'bwt':float(np.mean([final[u]-a[u,u] for u in old])),
      'acquisition':float(np.mean(np.diag(a))),'final_old_task_accuracy':float(np.mean(final[:-1])),'final_new_task_accuracy':float(final[-1])}


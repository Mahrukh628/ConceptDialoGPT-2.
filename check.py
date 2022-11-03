import json
output={}
with open('/content/for checking better results/generated_res.txt') as f:
            
            ind=0   
            for idx, line in enumerate(f):
                #print(line)
                #if idx % 100000 == 0: print('read train file line %d' % idx)
                data=" ".join(json.loads(line)["res_text"])
                #print(json.loads(line)["res_text"])
                output.update({ind:{"question":"","answer":data}})
                #print(data)
                ind=ind+1
                #data_train.append(json.loads(line))
            #print(output)
            k=0
            with open('/content/drive/MyDrive/My_project/My/inference_output_2/data/testset.txt') as f2:
                for idx, line in enumerate(f2):
                    if k<ind:
                        data= " ".join(json.loads(line)["post"])
                        output[k]["question"]=data
                    k=k+1
            #print(output)
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/My_project/My/inference_output_2/Data.csv')
with open('hyp.txt', 'w') as the_file_hyp:
    with open('ref.txt', 'w') as the_file_ref:
    #the_file.write('Hello\n')
        for x in output:
            #print(output[x])
            if len(output[x]["question"])>0 and len(output[x]["answer"])>0:
                the_file_ref.write(output[x]["question"]+'\n')
                the_file_hyp.write(output[x]["answer"]+'\n')
                row1={"index":(x+2487+0),"season no.":1,"episode no.":1,"episode name":"inference","name":"John","line":output[x]["question"]}
                row2={"index":(x+2487+1),"season no.":1,"episode no.":1,"episode name":"inference","name":"Harry","line":output[x]["answer"]}
                df = df.append(row1, ignore_index=True)
                df = df.append(row2, ignore_index=True)
                print(x)
        df.to_csv("Fina_Data.csv")
            
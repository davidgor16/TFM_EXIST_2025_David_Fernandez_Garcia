'''
Created on 1 may 2024

@author: jcalbornoz
'''
from os import listdir
from os.path import isfile, join
import json
import jsonschema
from jsonschema import validate

ID= "id"
TEST_CASE="test_case"
VALUE = "value"
TASK1_1="task1_1"
TASK1_2="task1_2"
TASK1_3="task1_3"
TASK2_1="task2_1"
TASK2_2="task2_2"
TASK2_3="task2_3"
TASK3_1="task2_1"
TASK3_2="task2_2"
TASK3_3="task2_3"

LIST_LABELS_SUBTASK1=["NO", "YES"]
LIST_LABELS_SUBTASK2=["NO", "REPORTED", "JUDGEMENTAL", "DIRECT"]
LIST_LABELS_SUBTASK3=["NO", "IDEOLOGICAL-INEQUALITY", "STEREOTYPING-DOMINANCE", "MISOGYNY-NON-SEXUAL-VIOLENCE", "SEXUAL-VIOLENCE", "OBJECTIFICATION"]
LIST_LABELS_SUBTASK2_2=["NO", "JUDGEMENTAL", "DIRECT"]

FORMAT_JSON_SCHEMA= {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "test_case": {"type": "string"},
            "id":{"type": "string"},
            "value": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"},"minItems": 1},
                    {"type": "integer"},
                    {
                        "type": "object",
                        "patternProperties": {
                        "^.*$": {"type": "number"},    }
                    },
                ]
            },              
        },
        "required": ["test_case", "id", "value"],
        "additionalProperties": False
    },      
    
}

def read_files_by_task(path):
    return [f for f in listdir(path) if isfile(join(path, f))] 


def parser_json(path):
    data = None
    try:
        with open(path, 'r', encoding='utf-8') as f:                
            data = json.load(f) 
    except ValueError as e:
        print(e)
        return False
    
    try:
        validate(instance=data, schema=FORMAT_JSON_SCHEMA)
    except jsonschema.exceptions.ValidationError as e:
        print("Errors found in the file: ", path , "\n", e)             
        return False
    
    return True


def process_format_runs_by_task(task_folder):
    onlyfiles = read_files_by_task(task_folder)
    for file in onlyfiles:   
        print("******************************************************")
        print("Analizing file "+ file)
        run = task_folder + file      
        if not parser_json(run):
            continue
         
        with open(run, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            tasks= file.split("_")
            if len(tasks)!=5:
                print("Error, the name of the run ", run, "is incorrect. Please check the Lab guidelines.")
                continue
            task= tasks[0]
            data_es=dict()
            data_en=dict()
            #Check size test data
            if task==TASK1_1 or task==TASK1_2 or task==TASK1_3:
                if len(data)!= 2076:
                    print("Error in the number of instances in Tweets test set: ", len(data), " . There must be 2076 instances.")
            elif task==TASK2_1 or task==TASK2_2 or task==TASK2_3:
                if len(data)!= 1053:
                    print("Error in the number of instances in Memes test set: ", len(data), " . There must be 1053 instances.")     
            elif task==TASK3_1 or task==TASK3_2 or task==TASK3_3:
                if len(data)!= 674:
                    print("Error in the number of instances in Videos test set: ", len(data), " . There must be 674 instances.")          

            for instance in data:              
                
                for property in instance:
                    #Check proporties in json object
                    if property==ID or property==TEST_CASE:
                        continue
                   
                    elif property==VALUE: 
                        if type(instance[VALUE])==type(""):                       
                            #Check labels
                            if task==TASK1_1 or task==TASK2_1 or task==TASK3_1:
                                if not instance[VALUE] in LIST_LABELS_SUBTASK1:
                                    print("ERROR in label for task ", task, " format ", VALUE, " hard label ", instance[VALUE])
                            
                            elif task==TASK1_2:
                                if not instance[VALUE] in LIST_LABELS_SUBTASK2:
                                    print("ERROR in label for task ", task, " format ", VALUE, " hard  label ", instance[VALUE])
                                    
                            elif task==TASK2_2 or task==TASK3_2:
                                if not instance[VALUE] in LIST_LABELS_SUBTASK2_2:
                                    print("ERROR in label for task ", task, " format ", VALUE, " hard  label ", instance[VALUE])
                                    
                        elif type(instance[VALUE])==type([]):
                            if task==TASK1_3 or task==TASK2_3 or task==TASK3_3 :
                                labels = instance[VALUE]
                                for label in labels:
                                    if not label in LIST_LABELS_SUBTASK3:
                                        print("ERROR in label for task ", task, " format ", VALUE, " hard  label ", instance[VALUE])                                          
                                                                    
                        elif type(instance[VALUE])==type(dict()):
                            if task==TASK1_1 or task==TASK2_1 or task==TASK3_1:
                                labels = instance[VALUE]
                                if len(labels)!=2:
                                    print("ERROR in label for task ", task, " format ", VALUE, " wrong number ", len(labels))
                                x=0
                                for label in labels:
                                    x+=float(labels[label])
                                    if not label in LIST_LABELS_SUBTASK1:
                                        print("ERROR in label for task ", task, " format ", VALUE, "soft label ", instance[VALUE])  
                                if x>1.001:
                                    print("ERROR in label for task ", task, " format ", VALUE, " sum is bigger than 1.0 ", x)       
                                    
                            elif task==TASK1_2:
                                labels = instance[VALUE]
                                if len(labels)!=4:
                                    print("ERROR in label for task ", task, " format ", VALUE, " wrong number ", len(labels))
                                x=0
                                for label in labels:
                                    x+=float(labels[label])
                                    if not label in LIST_LABELS_SUBTASK2:
                                        print("ERROR in label for task ", task, " format ", VALUE, "soft  label ", instance[VALUE])
                                if x>1.001:
                                    print("ERROR in label for task ", task, " format ", VALUE, " sum is bigger than 1.0 ", x)  
                                                                            
                            elif task==TASK2_2 or task==TASK3_2:
                                labels = instance[VALUE]
                                if len(labels)!=3:
                                    print("ERROR in label for task ", task, " format ", VALUE, " wrong number ", len(labels))
                                x=0
                                for label in labels:
                                    x+=float(labels[label])
                                    if not label in LIST_LABELS_SUBTASK2_2:
                                        print("ERROR in label for task ", task, " format ", VALUE, "soft  label ", instance[VALUE])
                                if x>1.001:
                                    print("ERROR in label for task ", task, " format ", VALUE, " sum is bigger than 1.0 ", x) 
                            
                            elif task==TASK1_3 or task==TASK2_3 or task==TASK3_3 :
                                labels = instance[VALUE]
                                if len(labels)!=6:
                                    print("ERROR in label for task ", task, " format ", VALUE, " wrong number ", len(labels))
                                for label in labels:
                                    if not label in LIST_LABELS_SUBTASK3:
                                        print("ERROR in label for task ", task, " format ", VALUE, "soft  label ", instance[VALUE])    
                        else:
                            print("Error format value property.")
                    
                    else:
                        print("ERROR in json format , property not allowed ", property)                                   
                        
                
                if task==TASK1_1 or task==TASK1_2 or task==TASK1_3:        
                    if int(instance[ID])>=500001 and int(instance[ID])<=501098:
                        data_es[instance[ID]]=instance   
                    elif int(instance[ID])>=600001 and int(instance[ID])<=600978: 
                        data_en[instance[ID]]=instance    
                    else:
                        print("ERROR en ids ", id)    
                elif task==TASK2_1 or task==TASK2_2 or task==TASK2_3:  
                    if int(instance[ID])>=310001 and int(instance[ID])<=310540:
                        data_es[instance[ID]]=instance   
                    elif int(instance[ID])>=410001 and int(instance[ID])<=410513: 
                        data_en[instance[ID]]=instance    
                    else:
                        print("ERROR en ids ", instance[ID])    
                elif task==TASK3_1 or task==TASK3_2 or task==TASK3_3:  
                    if int(instance[ID])>=320001 and int(instance[ID])<=320304:
                        data_es[instance[ID]]=instance   
                    elif int(instance[ID])>=420001 and int(instance[ID])<=420370: 
                        data_en[instance[ID]]=instance    
                    else:
                        print("ERROR en ids ", instance[ID])           

                       
            if task==TASK1_1 or task==TASK1_2 or task==TASK1_3: 
                if len(data_es)!=1098:
                    print("ERROR number of instances in the Tweets Spanish dataset ", len(data_es))
                if len(data_en)!=978:
                    print("ERROR  number of instances in the Tweets English dataset ", len(data_en)) 
            elif task==TASK2_1 or task==TASK2_2 or task==TASK2_2:               
                if len(data_es)!=540:
                    print("ERROR  number of instances in the Memes Spanish dataset ", len(data_es))
                if len(data_en)!=513:
                    print("ERROR  number of instances in the Memes English dataset ", len(data_en))             
            elif task==TASK3_1 or task==TASK3_2 or task==TASK3_2:               
                if len(data_es)!=304:
                    print("ERROR  number of instances in the Videos Spanish dataset ", len(data_es))
                if len(data_en)!=370:
                    print("ERROR  number of instances in the Videos English dataset ", len(data_en)) 
            
            print("******************************************************")
            
            
if __name__ == '__main__': 
    process_format_runs_by_task("dataset/2025/golds/")         
            
            
            
            
            
            
            
            
            
            
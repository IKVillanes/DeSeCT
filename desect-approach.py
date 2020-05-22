#!/usr/bin/env python
# coding: utf-8

# USING DEAP FRAMEWORK and NSGA2 TO SELECT DEVICES
import random
import pandas as pd
import numpy as np
from deap import base
from deap import creator
from deap import algorithms
from deap import tools
import matplotlib.pyplot as plt
import xlsxwriter
from collections import Counter
import os
import seaborn as sns
from datetime import datetime

#Time Starting
tstart = datetime.now()

# DataSet Location:
dir_file = os.path.dirname(__file__)
file_path = os.path.join(dir_file, '../Dataset/all_files_2559_9mixedDadosAPI_v2.xlsx')
dadosAll = pd.read_excel( file_path )

#Declaration if variables
resolucoes = []
versoes = []
redes = []
apiversion = []
screenvalor = []
dpivalor = []
brand = []
sensors = []


###### APP INFORMATION - PURPOSE  ######  
#EASY TAXY
lista_AppCar = {
        'app_sdk_min':'19',
        'app_sdk_max':'23',
        'ScreensSmall':'true',
        'ScreensNormal':'true',
        'ScreensLarge':'true',
        'ScreensXlarge':'true',
        'ScreensAnyDensity':'true'
        }
  
#NOT using  app information
#Deleting data from wereables devices API=0
#dados = dadosAll[dadosAll.APIversion != 0]

#Using app information to filter the dataset
dados = dadosAll.loc[dadosAll['APIversion'] >= int(lista_AppCar['app_sdk_min'])]  
dados = dados.loc[(dados['ScreenValor'].isin(['normal','small','large','xlarge']))]
   
#Counting UNIQUE values by features in dataset. This features will be used for coverage
resPxUniqueCount = len(dados["ResolPix"].value_counts())
sizeUniqueCount  = len(dados["Size"].value_counts())
apiUniqueCount = len(dados["APIversion"].value_counts())
versaoUniqueCount = len(dados["OSversion"].value_counts())
#List of features of all Dataset
resolucoes = dados["ResolPix"].tolist()
tamanhos = dados["Size"].tolist()
versoes = dados["OSversion"].tolist()
redes = dados["Technology"].tolist()
apiversion = dados["APIversion"].tolist()
screenvalor = dados["ScreenValor"].tolist()
dpivalor = dados["dpiValorTxt"].tolist()
brand = dados["Brand"].tolist()
sensors = dados["Sensors"].tolist()

#Variables to save coverage
cobertura_resolucoes = []
cobertura_tamanhos = []
cobertura_versoes = []
cobertura_redes = []
cobertura_screen = [] #Tuple(screen,dpi-valor)
cobertura_apiversion = []
cobertura_api = [] 
porcentage_resol = 0
porcentage_tam = 0
porcentage_ver = 0

#FEATURES DATA OBJECTIVE and WEIGHT
#Regras de Cobertura por caracteristica
list_porcent = [0.7, 0.7, 0.7, 0.7, 0.6]
#list_porcent = [resol, tam, api, redes, normal] 7  7  8   7  61
#Pesos atribuidos a cada caracteristica        
list_pesos = [0.20, 0.20, 0.30, 0.10, 0.20]
#list_pesos =[resol, tam, api, redes, normal]
#Densities
list_densities = ['ldpi', 'mdpi', 'hdpi', 'xhdpi', 'xxhdpi', 'xxxhdpi']
#Lista de redes
list_redes = ['2G', '3G', '4G', '5G']
#Usado para contabilizar cobertura de caracteristicas
quant_caract = len(list_pesos)
#Dictionary: Dados de redes e sus correspondencias #"No cellular connectivity":"NO"
dic_redes = {'GSM':"2G", 'CDMA':"2G", 'HSPA':"3G", 'UMTS':"3G", 'EVDO':"3G", 'CDMA2000':"3G", 'LTE':"4G", '5G':"5G" }
#Dictionary: Market share Android may 2019
ms_small = {'ldpi':"0.4", 'mdpi':"0", 'hdpi':"0", 'xhdpi':"0.1", 'xxhdpi':"0.1", 'xxxhdpi':"0"}
ms_normal = {'ldpi':"0", 'mdpi':"0.9", 'hdpi':"24.0", 'xhdpi':"37.7", 'xxhdpi':"23.6", 'xxxhdpi':"0"}
ms_large = {'ldpi':"0", 'mdpi':"2.4", 'hdpi':"0.6", 'xhdpi':"1.6", 'xxhdpi':"1.7", 'xxxhdpi':"0"}
ms_xlarge = {'ldpi':"0.4", 'mdpi':"3.1", 'hdpi':"1.3", 'xhdpi':"0.6", 'xxhdpi':"0", 'xxxhdpi':"0"}



#generating an Individual with limited number of 1's
def genFunkyInd(icls, size, maxone):
    ind_list = list()
    for i in range(size):
        ind_list.append("0")
    for i in range(maxone):
        index = np.random.randint(0, size - 1)
        ind_list[index] = "1"
    return icls(ind_list)

#new variable to selected device
#var_one = 1   #use for original individual
var_one = '1'   #use for modified individual

# Using  DEAP Framework
#using Creator for MultiObjective Fitness  (-minimization, +maximization)
creator.create("FitnessMulti", base.Fitness, weights=(1.0,-1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)
#Toolbox
toolbox = base.Toolbox()
#Chromossome with 1 and 0 indicating if was selected or not selected
toolbox.register("attr_bool", random.randint, 0, 1)

#Chromossome size equal to dataset size
#New invividual generation with limit of devices selected
max_ones = 250   #number of maximum devices selected at first
size_cromossome = len(dados)
toolbox.register('individual', genFunkyInd, creator.Individual, size_cromossome, max_ones)

#original way to generate invididual aprox 1000
#toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(dados))

#generating the population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#FUNCTION EVALUATION
## Function for get a total of devices selected - Minimization
def count_devices(individual):
    a = dict(Counter(individual))
    try:
        devices_selected = a[var_one]
    except Exception as e:
        devices_selected = 2000 #TODO ERRO SOMETIMES
    
    return devices_selected

#Function to calculate market share
def calcmarketshare(tuples_todos):
    sum_Small, sum_Normal, sum_Large, sum_Xlarge = 0, 0, 0, 0
    cov_ms_Small = set([(y) for (x,y) in tuples_todos if x == "small"])
    cov_ms_Normal = set([(y) for (x,y) in tuples_todos if x == "normal"])
    cov_ms_Large = set([(y) for (x,y) in tuples_todos if x == "large"])
    cov_ms_Xlarge = set([(y) for (x,y) in tuples_todos if x == "xlarge"])  
    
    for item in cov_ms_Small:
        for key, val in ms_small.items():
            if item == key: sum_Small += float(val)
                
    for item in cov_ms_Normal:
        for key, val in ms_normal.items():
            if item == key: sum_Normal += float(val)
                
    for item in cov_ms_Large:
        for key, val in ms_large.items():
            if item == key: sum_Large += float(val)
                
    for item in cov_ms_Xlarge:
        for key, val in ms_xlarge.items():
            if item == key: sum_Xlarge += float(val)     
    
    return(sum_Small, sum_Normal, sum_Large, sum_Xlarge)





## Function for get a features covered - maximization
def count_coverage(individual):  
    sizes_cantidade = len(list_densities)
    del cobertura_resolucoes[:]
    del cobertura_tamanhos[:]
    del cobertura_api[:]
    del cobertura_redes[:]
    del cobertura_screen[:]
   
    for i in range(len(individual)):
        #if individual[i] == int('1'):   #ik: original individual
        if individual[i] == var_one:   #ik: modified individual
            cobertura_resolucoes.append(resolucoes[i])
            cobertura_tamanhos.append(tamanhos[i])
            cobertura_api.append(apiversion[i]) if not np.isnan(apiversion[i]) else False #Some api data are empty on old devices
            cobertura_redes.append(redes[i])
#             cobertura_versoes.append(versoes[i])
            cobertura_screen.append((screenvalor[i],dpivalor[i]))
                 
#(Technology) network coverage has several other networks on the same device 
    lista_redes_cov = []
    for tech in cobertura_redes:
        for key, val in dic_redes.items():
            if key in tech:
                lista_redes_cov.append(val)
    conta_redes = len(set(lista_redes_cov))
    
#Calculating the coverage percentage obtained for each feature : selected_options/all_unique_options (result value between 0 -- 1)
    porcentage_resol = round(len(set(cobertura_resolucoes))/resPxUniqueCount,3)
    porcentage_tam = round(len(set(cobertura_tamanhos))/sizeUniqueCount,3)
    porcentage_api = round(len(set(cobertura_api))/apiUniqueCount,3)
    porcentage_redes = round(conta_redes/len(list_redes),3)
    
    tuples_todos = cobertura_screen
    res_marketshare = calcmarketshare(tuples_todos)
#"normal" density coverage, normal is the most used by Android devices
    #density_Normal = round(len(set([(y) for (x,y) in tuples_todos if x == "normal"]))/sizes_cantidade,3)
    return(porcentage_resol,porcentage_tam,porcentage_api,porcentage_redes,res_marketshare)


## Function for Fittness 
def eval_func(individual):
    cov_count=0
    sum_fitness=0
    dev_fitness = 1
    # 1: devices seleted -> minimun
    c_devices = count_devices(individual)
    
    # 2: features coverage -> maximum
    c_coverage_all = count_coverage(individual)
    c_coverage = c_coverage_all[:4] #features coverage
    c_coverage_market = c_coverage_all[-1]  #market share coverage  

    market_porcent = round(sum(c_coverage_market)/100,3)
    c_coverage = c_coverage + (market_porcent,)      
    
    #Calc All coverages
    for cov_res, cov_obj in zip(c_coverage, list_porcent):
       # print("resultado = %f   objetivo = %f" % (cov_res, cov_obj))
        if cov_res >= cov_obj: cov_count +=1
            
    
    #NEW PURPOSE WITHOUT DEVICE LIMITS
    #quant_caract = number of features that must be covered
    if cov_count >= quant_caract: 
        dev_fitness = c_devices
    else:
        dev_fitness = c_devices
        #dev_fitness = 3000 #(To review)
    
    #calc Fitness Coverage            
    for pesos, cov_res in zip(list_pesos, c_coverage):
        sum_fitness += pesos*cov_res
     
    return (sum_fitness, dev_fitness)


#Register fitness function
toolbox.register("evaluate", eval_func)
#Register the selection mode NSGA2
toolbox.register("select", tools.selNSGA2)
#Register the type of crossOver
toolbox.register("mate", tools.cxTwoPoint)
#Register the mutation and and 1% probability of each bit of the cromosome could be changed 
toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)

POP_SIZE = 200    # POP_SIZE = Population size
MAX_GEN = 10000  # MAX_GEN = maximum number of generations
MUT_PROB = 0.001  # MUT_PROB = Probability of mutation that each indivual of the population could be mutated
CX_PROB = 0.8     # CX_PROB = Probability of CrossOver  

pop = toolbox.population(n=POP_SIZE)   #Start initial population
hof = tools.HallOfFame(1)    #Instance the first best individual

# Register of Estatistics
stats = tools.Statistics(lambda ind: ind.fitness.values)
#stats.register("avg", np.mean) #, axis=0
stats.register("max", np.max)
stats.register("min", np.min)



# Executing the algorithm
result, log = algorithms.eaMuPlusLambda(pop, 
                                     toolbox, 
                                     mu=POP_SIZE,       # The number of individuals to select for the next generation.
                                     lambda_= POP_SIZE, # The number of children to produce at each generation.
                                     cxpb= CX_PROB,
                                     mutpb= MUT_PROB, 
                                     stats= stats, 
                                     ngen= MAX_GEN,
                                     halloffame=hof,
                                     verbose= True)

#Selecting One best Individual, could be two, three, etc best.
# melhores = tools.selBest(result, 1)
# for individuo in melhores: #or
# for individuo in hof:
#     print("Fitness value: ", individuo.fitness)


#All Pareto fronts results
# fronts = tools.emo.sortLogNondominated(result, len(result))
# resultado = fronts[0][0]

#Return ID data from select devices
lista_ids = []
#OPTION 1
for i in range(len(dados)):
    #if hof[0][i] == int('1'):   #ik: original individual
    if hof[0][i] == var_one:    #ik: modified individual      
        lista_ids.append(dados["ID"].values[i])

#OPTION 2
# for i in range(len(dados)):
#     if resultado[i] == 1:          
#         lista_ids.append(dados["ID"].values[i])

#Return information data from select devices
resultadoFinal = dados.loc[(dados['ID'].isin(lista_ids))]
print("Number of selected devices: ", len(resultadoFinal))
resultadoFinal.to_excel(r'resultado_comAppInfo10milGen-250.xlsx') #Save result in a file

#Deleting devices with same features ('APIversion', 'ScreenValor', 'dpiValorTxt')
resultadoFinal_noDuplicate = resultadoFinal.drop_duplicates(subset=['APIversion', 'ScreenValor', 'dpiValorTxt'],keep='last')
print("Number of selected devices (after filter): ", len(resultadoFinal_noDuplicate))
#resultadoFinal.to_excel(r'resultado_final_noDuplicate.xlsx')


#Fittness value from best indivual
print("Fittness Best Individual: ", eval_func(hof[0]))
coverage_individual = count_coverage(hof[0])
print("Coverage Individual: ", coverage_individual)

result_market_coverage = coverage_individual[-1]  #market share coverage  
sum_market_coverage = round(sum(result_market_coverage),2)
print("Sum market share: ", sum_market_coverage)

#CALCULATE SIMPSON INDEX FOR FEATURES
#Geting result data from features
simpson_Resolution = (resultadoFinal["ResolPix"].value_counts())
simpson_Size = (resultadoFinal["Size"].value_counts())
simpson_APIversion = (resultadoFinal["APIversion"].value_counts())


#Function to calculate simpson index
def simpsonindex(data):
    N = sum(data)
    s = 0
    for item in data:
        s +=(item*(item-1))
    simpson_value = 1-(s/(N*(N-1)))
    return simpson_value

print("Simpson index RESOLUTION:", simpsonindex(simpson_Resolution))
print("Simpson index SIZE:", simpsonindex(simpson_Size))
print("Simpson index API:", simpsonindex(simpson_APIversion))

#Time Ending
tend = datetime.now()

print("Tempo de execucao:", tend-tstart)

#Ploting results
valores_grafico = log.select("min")
sns.set_style("whitegrid")
plt.plot(valores_grafico, color='red')
plt.title("Maximization - Fittness value")
plt.show()


valores_grafico2 = log.select("max")
sns.set_style("whitegrid")
plt.plot(valores_grafico2, color='blue')
plt.title("Minimization - Fittnes value")
plt.show()



'''minFitnessValues, meanFitnessValues = log.select("min", "avg")
sns.set_style("whitegrid")
plt.plot(minFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Generation')
plt.ylabel('Min / Average Fitness')
plt.title('Min and Average fitness over Generations')
plt.show()'''